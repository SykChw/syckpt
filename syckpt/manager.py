"""Main CheckpointManager - Git-like checkpoint system with LSH hashing and CAS storage."""

import os
import json
import fcntl
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from syckpt.config import HyperConfig
from syckpt.hash import LSHHashGenerator
from syckpt.state import (
    StateManager,
    get_rng_state,
    set_rng_state,
    get_deterministic_state,
    set_deterministic_state,
)
from syckpt.storage import CASStorage, flatten_state, unflatten_state

logger = logging.getLogger(__name__)


class Lock:
    """File-based lock using fcntl for distributed safety."""

    __slots__ = ("lock_path", "timeout", "_fd")

    def __init__(self, lock_path: Path, timeout: int = 30):
        self.lock_path = lock_path
        self.timeout = timeout
        self._fd = None

    def acquire(self) -> bool:
        import time

        start = time.time()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                self._fd = open(self.lock_path, "w")
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd.write(str(os.getpid()))
                self._fd.flush()
                return True
            except (IOError, OSError):
                if self._fd:
                    self._fd.close()
                if time.time() - start > self.timeout:
                    return False
                time.sleep(0.1)
        return False

    def release(self):
        if self._fd:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                self._fd.close()
            except Exception:
                pass
            finally:
                self._fd = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"Could not acquire lock: {self.lock_path}")
        return self

    def __exit__(self, *args):
        self.release()
        return False


class Commit:
    """Represents a saved checkpoint (like git commit)."""

    __slots__ = (
        "hash",
        "parent",
        "message",
        "step",
        "epoch",
        "config",
        "metric",
        "timestamp",
        "blob_hash",
        "blob_metadata",
        "components_structure",
        "rng",
        "deterministic"
    )

    def __init__(
        self,
        hash: str,
        parent: Optional[str] = None,
        message: str = "",
        step: int = 0,
        epoch: int = 0,
        config: Optional[Dict] = None,
        metric: Optional[float] = None,
        blob_hash: Optional[str] = None,
        blob_metadata: Optional[Dict] = None,
        components_structure: Optional[Dict] = None,
        rng: Optional[Any] = None,
        deterministic: Optional[Any] = None,
    ):
        self.hash = hash
        self.parent = parent
        self.message = message
        self.step = step
        self.epoch = epoch
        self.config = config or {}
        self.metric = metric
        self.timestamp = datetime.now().isoformat()
        self.blob_hash = blob_hash or hash
        self.blob_metadata = blob_metadata or {}
        self.components_structure = components_structure
        self.rng = rng
        self.deterministic = deterministic

    def to_dict(self) -> Dict:
        return {
            "hash": self.hash,
            "parent": self.parent,
            "message": self.message,
            "step": self.step,
            "epoch": self.epoch,
            "config": self.config,
            "metric": self.metric,
            "timestamp": self.timestamp,
            "blob_hash": self.blob_hash,
            "blob_metadata": self.blob_metadata,
            "components_structure": self.components_structure,
            "rng": self.rng,
            "deterministic": self.deterministic
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Commit":
        c = cls(
            data["hash"],
            data.get("parent"),
            data.get("message", ""),
            data.get("step", 0),
            data.get("epoch", 0),
            data.get("config", {}),
            data.get("metric"),
            data.get("blob_hash"),
            data.get("blob_metadata", {}),
            data.get("components_structure"),
            data.get("rng"),
            data.get("deterministic")
        )
        c.timestamp = data.get("timestamp", c.timestamp)
        return c


class CheckpointManager:
    """Git-like checkpoint manager with LSH hashing and CAS Storage natively handling safetensors and fsspec."""

    __slots__ = (
        "root",
        "storage",
        "max_to_keep",
        "maximize",
        "auto_resume",
        "save_rng",
        "_lock",
        "_locked",
        "state_manager",
        "_config",
        "_step",
        "_epoch",
        "_batch_idx",
        "_hash",
        "_current_branch",
        "_commits",
        "_session_commits",
        "_session_start_hash",
        "_top_k_metrics",
        "run_mode",
        "_bg_processes",
    )

    def __init__(
        self,
        dirpath: Union[str, Path],
        max_to_keep: int = 5,
        maximize: bool = False,
        auto_resume: bool = True,
        save_rng: bool = True,
        lock_timeout: int = 30,
        hash_length: int = 8,
        run_mode: str = "new_branch",
    ):
        self.root = str(dirpath)
        self.storage = CASStorage(self.root)
        
        self.max_to_keep = max_to_keep
        self.maximize = maximize
        self.auto_resume = auto_resume
        self.save_rng = save_rng
        self.run_mode = run_mode

        # Lock is only used if the path is primarily local
        if not self.root.startswith(("s3://", "gcs://", "http://", "https://")):
            lock_path = Path(self.storage.fs_path) / ".syckpt" / ".lock"
            self._lock = Lock(lock_path, lock_timeout)
        else:
            self._lock = None
            
        self._locked = False

        self.state_manager = StateManager()
        self._config = HyperConfig()
        self._step = 0
        self._epoch = 0
        self._batch_idx = 0

        self._hash: Optional[str] = None
        self._current_branch = self.storage.read_head()
        self._commits: Dict[str, Commit] = {}
        self._session_commits: List[str] = []
        self._session_start_hash: Optional[str] = None
        self._top_k_metrics: List[Tuple[float, str]] = []
        self._bg_processes: list = []

        # Load latest commit from the active branch if it exists.
        # Skip mega-hash tips — they have no tensor blob and exist only as UI containers.
        branch_hash = self.storage.read_ref(self._current_branch)
        if branch_hash and self.storage.check_commit_exists(branch_hash):
            self._hash = branch_hash
            try:
                data = self.storage.load_commit(branch_hash)
                if not data.get("is_mega"):
                    self._load_commit_into_cache(branch_hash)
            except Exception:
                pass

    # Properties
    @property
    def config(self) -> HyperConfig:
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, HyperConfig):
            self._config = value
        elif isinstance(value, dict):
            self._config = HyperConfig(value)

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int):
        self._step = value

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int):
        self._epoch = value

    @property
    def batch_idx(self) -> int:
        return self._batch_idx

    @batch_idx.setter
    def batch_idx(self, value: int):
        self._batch_idx = value

    @property
    def hash(self) -> str:
        return self._hash or "uninitialized"

    @property
    def branch(self) -> str:
        return self._current_branch

    # Registration
    def register(self, **kwargs):
        self.state_manager.register(**kwargs)

    def unregister(self, *names):
        self.state_manager.unregister(*names)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        comp = self.state_manager.get(name)
        if comp is not None:
            return comp
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_") or name in (
            "config",
            "step",
            "epoch",
            "batch_idx",
            "hash",
            "branch",
            "storage",
            "root",
            "max_to_keep",
            "maximize",
            "auto_resume",
            "save_rng",
            "state_manager",
            "run_mode"
        ):
            object.__setattr__(self, name, value)
        else:
            self.state_manager.register(**{name: value})

    # Locking
    def _lock_acquire(self):
        if self._lock and not self._locked:
            self._lock.__enter__()
            self._locked = True

    def _lock_release(self):
        if self._lock and self._locked:
            self._lock.__exit__(None, None, None)
            self._locked = False

    # Hash generation
    def _generate_hash(self) -> str:
        gen = LSHHashGenerator(hash_length=8)
        config_dict = self._config.to_dict() if self._config else {}

        components = {}
        for name in self.state_manager.list_components():
            components[name] = self.state_manager.get(name)

        return gen.generate_from_components(
            config_dict, components.get("model"), components.get("optimizer")
        )

    # Core State Encoding/Decoding
    def _build_commit_data(self) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        components_state = self.state_manager.build_state()
        metadata_structure, flat_tensors = flatten_state(components_state)
        
        metadata = {
            "step": self._step,
            "epoch": self._epoch,
            "batch_idx": self._batch_idx,
            "timestamp": datetime.now().isoformat(),
            "branch": self._current_branch,
            "config": self._config.to_dict() if self._config else {},
            "components_structure": metadata_structure,
            "rng": get_rng_state() if self.save_rng else None,
            "deterministic": get_deterministic_state(),
        }
        return metadata, flat_tensors
        
    def _restore_commit_data(self, metadata: Dict[str, Any], flat_tensors: Dict[str, torch.Tensor]):
        self._step = metadata.get("step", 0)
        self._epoch = metadata.get("epoch", 0)
        self._batch_idx = metadata.get("batch_idx", 0)
        self._current_branch = metadata.get("branch", "main")
        
        self._config = HyperConfig.from_dict(metadata.get("config", {}))
        
        if "components_structure" in metadata:
            components_state = unflatten_state(metadata["components_structure"], flat_tensors)
            self.state_manager.restore_state(components_state)
            
        if self.save_rng and "rng" in metadata and metadata["rng"]:
            rng_state = metadata["rng"]
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and isinstance(rng_state, list):
                rank = dist.get_rank()
                if rank < len(rng_state):
                    set_rng_state(rng_state[rank])
                else:
                    set_rng_state(rng_state[0])
            else:
                if isinstance(rng_state, list):
                    set_rng_state(rng_state[0])
                else:
                    set_rng_state(rng_state)
            
        if "deterministic" in metadata:
            set_deterministic_state(metadata["deterministic"])

    def _fetch_tensors(self, commit_hash: str) -> Dict[str, torch.Tensor]:
        """Recursively resolves delta-compressed safetensors blobs for a commit."""
        commit_data = self.storage.load_commit(commit_hash)
        
        if commit_data.get("is_mega") and commit_data.get("sub_commits"):
            return self._fetch_tensors(commit_data["sub_commits"][-1])
            
        blob_metadata = commit_data.get("blob_metadata", {})
        blob_hash = blob_metadata.get("blob_hash", commit_hash)
        
        if blob_metadata.get("is_delta"):
            base_hash = commit_data.get("parent")
            if not base_hash:
                raise ValueError(f"Commit {commit_hash} requires delta resolution but has no parent.")
            if base_hash == commit_hash:
                raise ValueError(f"Commit {commit_hash} points to itself as a parent, infinite loop detected.")
            
            base_tensors = self._fetch_tensors(base_hash)
            return self.storage.load_tensors(
                blob_hash, 
                base_tensors=base_tensors, 
                is_delta=True, 
                frozen_links=blob_metadata.get("frozen_links", {})
            )
        else:
            return self.storage.load_tensors(blob_hash, is_delta=False)

    def _load_commit_into_cache(self, commit_hash: str):
        if commit_hash not in self._commits:
            data = self.storage.load_commit(commit_hash)
            self._commits[commit_hash] = Commit.from_dict(data)

    # Branch operations
    def create_branch(self, name: str, message: str = "") -> str:
        self._lock_acquire()
        try:
            if self.storage.read_ref(name):
                self._current_branch = name
                self.storage.write_head(name)
                return name

            new_hash = self._generate_hash()
            commit_data = {
                "hash": new_hash,
                "parent": self._hash,
                "message": message or f"Created branch: {name}",
                "step": self._step,
                "epoch": self._epoch,
                "config": self._config.to_dict(),
            }
            
            self.storage.save_commit(new_hash, commit_data)
            self.storage.write_ref(name, new_hash)
            self.storage.write_head(name)
            
            self._current_branch = name
            self._hash = new_hash
            self._commits[new_hash] = Commit.from_dict(commit_data)

            logger.info(f"Created branch: {name}")
            return name
        finally:
            self._lock_release()

    def checkout_branch(self, name: str) -> bool:
        self._lock_acquire()
        try:
            target_hash = self.storage.read_ref(name)
            if not target_hash:
                raise ValueError(f"Branch not found: {name}")

            self._current_branch = name
            self.storage.write_head(name)
            
            if self.storage.check_commit_exists(target_hash):
                commit_data = self.storage.load_commit(target_hash)
                flat_tensors = self._fetch_tensors(target_hash)
                self._restore_commit_data(commit_data, flat_tensors)
                self._hash = target_hash

            logger.info(f"Switched to branch: {name}")
            return True
        finally:
            self._lock_release()

    def branch(self, message: str = "", **hyperparams) -> str:
        self._lock_acquire()
        try:
            for key, value in hyperparams.items():
                self._config[key] = value

            new_hash = self._generate_hash()

            if new_hash != self._hash:
                # Need to act like a branch creation but update current
                commit_data = {
                    "hash": new_hash,
                    "parent": self._hash,
                    "message": message or f"Branch: {', '.join(hyperparams.keys())}",
                    "step": self._step,
                    "epoch": self._epoch,
                    "config": self._config.to_dict(),
                }
                self.storage.save_commit(new_hash, commit_data)
                self.storage.write_ref(self._current_branch, new_hash)
                self._hash = new_hash
                self._commits[new_hash] = Commit.from_dict(commit_data)

            return self._hash
        finally:
            self._lock_release()

    def goto(self, hash_or_branch: str) -> bool:
        self._lock_acquire()
        try:
            if self.storage.read_ref(hash_or_branch):
                return self.checkout_branch(hash_or_branch)

            if not self.storage.check_commit_exists(hash_or_branch):
                raise FileNotFoundError(f"Commit not found: {hash_or_branch}")

            commit_data = self.storage.load_commit(hash_or_branch)
            flat_tensors = self._fetch_tensors(hash_or_branch)
            self._restore_commit_data(commit_data, flat_tensors)
            self._hash = hash_or_branch

            return True
        finally:
            self._lock_release()

    def delete_branch(self, branch: str) -> bool:
        if branch == "main":
            raise ValueError("Cannot delete main branch")
            
        self._lock_acquire()
        try:
            success = self.storage.delete_ref(branch)
            if success:
                logger.info(f"Deleted branch: {branch}")
            return success
        finally:
            self._lock_release()

    def list_branches(self) -> list:
        return self.storage.list_branches()

    def log(self, n: int = 10) -> list:
        history = []
        current = self._hash
        while current and len(history) < n:
            if current not in self._commits:
                if self.storage.check_commit_exists(current):
                    self._load_commit_into_cache(current)
                else:
                    break
            history.append(self._commits[current])
            current = self._commits[current].parent
        return history

    def diff(self, hash1: str, hash2: str) -> Dict:
        c1_data = self.storage.load_commit(hash1)
        c2_data = self.storage.load_commit(hash2)
        
        c1_config = c1_data.get("config", {})
        c2_config = c2_data.get("config", {})

        all_keys = set(c1_config.keys()) | set(c2_config.keys())
        diff = {"hash1": hash1[:8], "hash2": hash2[:8], "config_diff": {}}
        for k in all_keys:
            if c1_config.get(k) != c2_config.get(k):
                diff["config_diff"][k] = {
                    "v1": c1_config.get(k),
                    "v2": c2_config.get(k),
                }
        return diff

    # Save/Load - utilizing Safetensors & Delta Compression
    def _update_top_k(self, current_hash: str, metric: float):
        if self.max_to_keep <= 0:
            return
            
        self._top_k_metrics.append((metric, current_hash))
        self._top_k_metrics.sort(key=lambda x: x[0], reverse=self.maximize)
        
        if len(self._top_k_metrics) > self.max_to_keep:
            self._top_k_metrics = self._top_k_metrics[:self.max_to_keep]
            
        # Wipe all old tags and rewrite Best-K
        for tag in self.storage.list_tags():
            if tag.startswith("best_"):
                self.storage.delete_tag(tag)
                
        for i, (_, h) in enumerate(self._top_k_metrics):
            self.storage.write_tag(f"best_{i+1}", h)

    def save(self, metric: Optional[float] = None, message: str = "") -> str:
        self._lock_acquire()
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                is_main = dist.get_rank() == 0
                world_size = dist.get_world_size()
                is_dist = True
            else:
                is_main = True
                is_dist = False

            current_hash = self._generate_hash() if is_main else ""
            
            if is_dist:
                hash_list = [current_hash]
                dist.broadcast_object_list(hash_list, src=0)
                current_hash = hash_list[0]
                
            rng_state = get_rng_state() if self.save_rng else None
            
            if is_dist and self.save_rng:
                if is_main:
                    gathered_rngs = [None for _ in range(world_size)]
                    dist.gather_object(rng_state, gathered_rngs, dst=0)
                    rng_state = gathered_rngs
                else:
                    dist.gather_object(rng_state, dst=0)
                    
            if not is_main:
                if is_dist:
                    dist.barrier()
                self._hash = current_hash
                return current_hash

            base_hash = self._hash
            
            # Anti-Collision: LSH will output identical hashes for minor delta patches!
            base_c_hash = current_hash
            while self.storage.check_commit_exists(current_hash) or current_hash in self._commits:
                import uuid
                current_hash = f"{base_c_hash}-{uuid.uuid4().hex[:6]}"

            # If base_hash is the same as (or not yet a committed) current_hash, treat as root.
            # This happens for the very first save where __enter__ sets _hash = generated hash.
            if base_hash == current_hash or not self.storage.check_commit_exists(base_hash):
                base_hash = None
                
            metadata, flat_tensors = self._build_commit_data()
            
            if self.save_rng:
                metadata["rng"] = rng_state

            blob_hash = current_hash # using same naming for blob
            commit_data = {
                "hash": current_hash,
                "parent": base_hash,
                "message": message or "Update",
                "metric": metric,
                "blob_hash": blob_hash,
                **metadata
            }
            
            # Offload heavy IO and CPU memory allocation to a multiprocessing fork
            # We explicitly detach the tensors from the live GPU to CPU before handing off to the process.
            cpu_tensors = {k: v.to("cpu", non_blocking=True).clone() for k, v in flat_tensors.items()}
            
            def _async_save_worker(comp_tensors, c_hash, b_hash, c_data, b_branch, fs_storage):
                import logging
                logger = logging.getLogger(__name__)
                
                logger.info(f"Async Save Started [PID: {os.getpid()}] processing blob {c_hash}")
                b_tensors = None
                if b_hash and fs_storage.check_commit_exists(b_hash):
                    # Skip mega-hash commits — they are UI containers with no tensor blob.
                    try:
                        b_meta = fs_storage.load_commit(b_hash)
                        if not b_meta.get("is_mega"):
                            b_tensors = fs_storage.load_tensors(b_hash, is_delta=False)
                    except Exception:
                        b_tensors = None

                blob_meta = fs_storage.save_tensors(comp_tensors, c_hash, base_tensors=b_tensors)
                c_data["blob_metadata"] = blob_meta

                fs_storage.save_commit(c_hash, c_data)
                fs_storage.write_ref(b_branch, c_hash)
                fs_storage.write_head(b_branch)
                logger.info(f"Async Save Finished [PID: {os.getpid()}]")

            import multiprocessing
            p = multiprocessing.Process(
                target=_async_save_worker,
                args=(cpu_tensors, current_hash, base_hash, commit_data, self._current_branch, self.storage)
            )
            p.start()
            self._bg_processes.append(p)

            # We locally update the host references instantly without waiting for the disk write!
            self._hash = current_hash
            self._commits[current_hash] = Commit.from_dict(commit_data)
            
            if is_main:
                self._session_commits.append(current_hash)
                if metric is not None:
                    self._update_top_k(current_hash, metric)
            
            # NOTE: We specifically do not call `dist.barrier()` here anymore.
            # If we called `dist.barrier()` at the end, Rank 0 would instantly hit the barrier
            # and unblock the workers while its background process was still writing! 
            # This is exactly what we want for pure async execution.
                
            return current_hash
        finally:
            self._lock_release()

    def load(self, hash: Optional[str] = None) -> Dict:
        self._lock_acquire()
        try:
            if hash is None:
                hash = self._hash
            
            if not hash:
                raise ValueError("No hash provided and manager is uninitialized.")

            if not self.storage.check_commit_exists(hash):
                raise ValueError(f"Commit not found: {hash}")

            commit_data = self.storage.load_commit(hash)

            # Mega-hash commits are UI grouping containers with no tensor blob.
            # Transparently resolve to the last real sub-commit.
            if commit_data.get("is_mega") and commit_data.get("sub_commits"):
                last_sub = commit_data["sub_commits"][-1]
                if self.storage.check_commit_exists(last_sub):
                    commit_data = self.storage.load_commit(last_sub)
                    hash = last_sub

            flat_tensors = self._fetch_tensors(hash)
            self._restore_commit_data(commit_data, flat_tensors)

            return {
                "hash": self._hash,
                "step": self._step,
                "epoch": self._epoch,
                "batch_idx": self._batch_idx,
                "config": self._config,
            }
        finally:
            self._lock_release()


    # load_into_* functions
    def load_into_model(self, model: nn.Module, hash: Optional[str] = None) -> None:
        self._lock_acquire()
        try:
            h = hash or self._hash
            commit_data = self.storage.load_commit(h)
            flat_tensors = self._fetch_tensors(h)
            components = unflatten_state(commit_data.get("components_structure", {}), flat_tensors)
            
            if "model" in components:
                model.load_state_dict(components["model"])
            else:
                raise ValueError("No model state in checkpoint")
        finally:
            self._lock_release()

    def load_into_optimizer(
        self, optimizer: torch.optim.Optimizer, hash: Optional[str] = None
    ) -> None:
        self._lock_acquire()
        try:
            h = hash or self._hash
            commit_data = self.storage.load_commit(h)
            flat_tensors = self._fetch_tensors(h)
            components = unflatten_state(commit_data.get("components_structure", {}), flat_tensors)

            if "optimizer" in components:
                optimizer.load_state_dict(components["optimizer"])
            else:
                raise ValueError("No optimizer state in checkpoint")
        finally:
            self._lock_release()

    def load_into_scheduler(self, scheduler, hash: Optional[str] = None) -> None:
        self._lock_acquire()
        try:
            h = hash or self._hash
            commit_data = self.storage.load_commit(h)
            flat_tensors = self._fetch_tensors(h)
            components = unflatten_state(commit_data.get("components_structure", {}), flat_tensors)

            if "scheduler" in components:
                scheduler.load_state_dict(components["scheduler"])
            else:
                raise ValueError("No scheduler state in checkpoint")
        finally:
            self._lock_release()

    def load_into_dataloader(
        self, dataloader: DataLoader, hash: Optional[str] = None
    ) -> int:
        self._lock_acquire()
        try:
            h = hash or self._hash
            commit_data = self.storage.load_commit(h)
            flat_tensors = self._fetch_tensors(h)
            components = unflatten_state(commit_data.get("components_structure", {}), flat_tensors)

            batch_idx = commit_data.get("batch_idx", 0)

            if "dataloader" in components:
                dl_state = components["dataloader"]
                if hasattr(dataloader, "load_state_dict"):
                    dataloader.load_state_dict(dl_state)

            return batch_idx
        finally:
            self._lock_release()

    def load_into_config(self, hash: Optional[str] = None) -> HyperConfig:
        h = hash or self._hash
        if not self.storage.check_commit_exists(h):
            raise ValueError(f"Commit not found: {h}")
            
        commit_data = self.storage.load_commit(h)
        return HyperConfig.from_dict(commit_data.get("config", {}))

    def load_all(self, hash: Optional[str] = None) -> Dict[str, Any]:
        self._lock_acquire()
        try:
            h = hash or self._hash
            commit_data = self.storage.load_commit(h)
            flat_tensors = self._fetch_tensors(h)
            self._restore_commit_data(commit_data, flat_tensors)
            
            return {
                "step": self._step,
                "epoch": self._epoch,
                "batch_idx": self._batch_idx,
                "config": self._config,
                "components": unflatten_state(commit_data.get("components_structure", {}), flat_tensors)
            }
        finally:
            self._lock_release()

    def export_ckpt(self, hash_or_branch: str, output_path: Union[str, Path]) -> None:
        """Exports a specific commit into a standard monolithic PyTorch .ckpt file."""
        self._lock_acquire()
        try:
            target_hash = self.storage.read_ref(hash_or_branch) or hash_or_branch
            if not target_hash or not self.storage.check_commit_exists(target_hash):
                raise ValueError(f"Commit not found: {hash_or_branch}")

            commit_data = self.storage.load_commit(target_hash)
            flat_tensors = self._fetch_tensors(target_hash)
            components = unflatten_state(commit_data.get("components_structure", {}), flat_tensors)
            
            monolithic_state = {
                "hash": target_hash,
                "step": commit_data.get("step", 0),
                "epoch": commit_data.get("epoch", 0),
                "batch_idx": commit_data.get("batch_idx", 0),
                "branch": commit_data.get("branch", "main"),
                "config": commit_data.get("config", {}),
                "components": components,
                "rng": commit_data.get("rng"),
                "deterministic": commit_data.get("deterministic")
            }
            
            # Using standard torch.save for ecosystem compatibility
            torch.save(monolithic_state, str(output_path))
            logger.info(f"Successfully exported {target_hash} to {output_path}")
        finally:
            self._lock_release()

    # Utilities
    def step_up(self):
        self._step += 1

    def step_to(self, step: int):
        self._step = step

    def loop(self, epochs: int, steps_per_epoch: Optional[int] = None):
        start = self._epoch
        self._session_start_hash = self._hash
        self._session_commits = []
        try:
            for ep in range(start, epochs):
                self._epoch = ep
                if steps_per_epoch is None:
                    yield ep
                else:
                    for st in range(steps_per_epoch):
                        self._step = ep * steps_per_epoch + st
                        yield ep, st
        finally:
            # Wait for all pending async workers before grouping \u2014 workers write the branch ref to
            # the individual sub-commit hash, so we must let them finish FIRST, then overwrite
            # that ref with the mega-hash.
            for p in self._bg_processes:
                p.join(timeout=120)
            self._bg_processes.clear()
            if self._session_commits:
                self.group_commits(message=f"Loop Mega-Hash ({epochs} epochs)")


    def list_checkpoints(self) -> Dict[str, str]:
        # Without branches folder per commit, we just list the branch tip commits
        return {branch: self.storage.read_ref(branch) for branch in self.storage.list_branches()}

    def commit_info(self) -> Optional[Commit]:
        if self._hash and self.storage.check_commit_exists(self._hash):
            return Commit.from_dict(self.storage.load_commit(self._hash))
        return None

    # Context manager
    def __enter__(self) -> "CheckpointManager":
        self._lock_acquire()

        self._session_commits = []

        if self.run_mode == "overwrite":
            # Delete current branch tip and start a completely fresh run on the same branch.
            self.storage.delete_ref(self._current_branch)
            self._hash = self._generate_hash()
            self._session_start_hash = self._hash
            logger.info(f"Overwrote branch {self._current_branch}, starting fresh.")

        elif self.run_mode == "new_branch":
            # Always fork a new branch so each run is completely independent.
            existing = self.storage.read_ref(self._current_branch)
            if existing:
                # Pre-load model weights from the last checkpoint so the new branch
                # starts with trained weights. Reset counters since this is a fresh run.
                try:
                    self.load(existing)
                except Exception as e:
                    logger.warning(f"Could not pre-load state from {existing}: {e}")
            # Reset epoch/step \u2014 new_branch is a fresh run, not a continuation.
            self._epoch = 0
            self._step = 0
            self._batch_idx = 0
            import uuid
            base = self._current_branch.split("_continue_")[0]
            new_branch_name = f"{base}_continue_{uuid.uuid4().hex[:4]}"
            self._current_branch = new_branch_name
            self.storage.write_head(new_branch_name)
            self._hash = self._generate_hash()
            self._session_start_hash = self._hash
            logger.info(f"Created new branch {new_branch_name} for this run.")

        else:  # append
            if self.auto_resume:
                latest = self.storage.read_ref(self._current_branch)
                if latest and self.storage.check_commit_exists(latest):
                    try:
                        self.load(latest)
                        logger.info(
                            f"Resumed from step {self._step}, batch {self._batch_idx} "
                            f"on branch {self._current_branch}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to resume from {latest}: {e}")
            if self._hash is None:
                self._hash = self._generate_hash()
            self._session_start_hash = self._hash

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self.save(message=f"[FAILED] \u274c {exc_type.__name__}")
                logger.error(f"Training failed with {exc_type.__name__}. Saved failure state.")
        except Exception as e:
            logger.warning(f"Failed to save on exit: {e}")
        finally:
            # NOTE: group_commits is already called by loop()'s finally block.
            # Only call it here if the user is NOT using ckpt.loop() (i.e. manual saves).
            if self._session_commits:
                self.group_commits(message="Training Loop Mega-Hash")
            # Wait for all async save workers to flush to disk before printing.
            for p in self._bg_processes:
                p.join(timeout=120)
            self._bg_processes.clear()
            self._lock_release()
            self.print_tree()
        return False

    def group_commits(self, message: str = "Mega-Hash"):
        """Clubs recent session commits into a single UI MegaCommit."""
        if not self._session_commits or len(self._session_commits) <= 1:
            return
            
        import uuid
        mega_hash = f"mega_{uuid.uuid4().hex[:8]}"
        last_commit = self._session_commits[-1]
        
        if last_commit not in self._commits:
            return
            
        last_data = self._commits[last_commit]
        
        mega_commit_data = {
            "hash": mega_hash,
            "parent": self._session_start_hash,
            "is_mega": True,
            "sub_commits": self._session_commits,
            "message": message,
            "step": self._step,
            "epoch": self._epoch,
            "metric": last_data.metric,
            "config": self._config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "components_structure": last_data.components_structure or {},
            "rng": last_data.rng,
            "deterministic": last_data.deterministic
        }
        
        self.storage.save_commit(mega_hash, mega_commit_data)
        self.storage.write_ref(self._current_branch, mega_hash)
        self.storage.write_head(self._current_branch)
        
        self._hash = mega_hash
        self._commits[mega_hash] = Commit.from_dict(mega_commit_data)
        self._session_commits = []

    def print_tree(self):
        """Prints the entire commit tree across all branches and highlights the current HEAD."""
        try:
            tree_data = self.storage.get_commit_tree()
        except Exception:
            return
            
        commits = tree_data["commits"]
        branch_tips = tree_data["branch_tips"]
        tags = tree_data.get("tags", {})

        # Collect all sub-commit hashes so we can hide them from the top-level view.
        # They will be shown inline under their parent mega-hash.
        sub_commit_set: set = set()
        for h, c in commits.items():
            if c.get("is_mega"):
                sub_commit_set.update(c.get("sub_commits", []))

        # Build parent → children map (excluding sub-commits from tree hierarchy)
        top_commits = {h: c for h, c in commits.items() if h not in sub_commit_set}
        children: dict = {h: [] for h in top_commits}

        roots = []
        for h, c in top_commits.items():
            p = c.get("parent")
            if not p or p not in top_commits or p == h:
                roots.append(h)
            else:
                children[p].append(h)

        def _print_node(node_hash, prefix="", is_last=True):
            c = top_commits[node_hash]

            is_mega = c.get("is_mega")
            if is_mega:
                sub_list = c.get("sub_commits", [])
                msg = f"[MEGA-HASH] {len(sub_list)} sub-commits | {c.get('message', '')}"
            else:
                msg = c.get("message", "")

            epoch = c.get("epoch", 0)
            metric = c.get("metric")
            metric_str = f" | metric: {metric:.4f}" if metric is not None else ""

            labels = []
            if node_hash == self._hash:
                labels.append("HEAD")
            for b, tip in branch_tips.items():
                if tip == node_hash:
                    labels.append(f"*{b}*" if b == self._current_branch else b)
            for t_name, t_val in tags.items():
                if t_val == node_hash:
                    labels.append(f"[TAG: {t_name}]")

            label_str = f" ({', '.join(labels)})" if labels else ""
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{node_hash[:8]}{label_str}: {msg} [Epoch {epoch}]{metric_str}")

            # If this is a mega-hash, list sub-commits inline with indentation
            if is_mega:
                sub_list = c.get("sub_commits", [])
                sub_prefix = prefix + ("    " if is_last else "│   ")
                for j, sub in enumerate(sub_list):
                    sub_c = commits.get(sub, {})
                    sub_msg = sub_c.get("message", "")
                    sub_ep = sub_c.get("epoch", "?")
                    sub_connector = "└── " if j == len(sub_list) - 1 else "├── "
                    print(f"{sub_prefix}{sub_connector}{sub[:8]}: {sub_msg} [Epoch {sub_ep}]")

            child_list = children.get(node_hash, [])
            for i, child in enumerate(child_list):
                extension = "    " if is_last else "│   "
                _print_node(child, prefix + extension, i == len(child_list) - 1)

        print(f"\n--- Syckpt Tree [{'LOCAL' if not self._lock else 'DIST'}] ---")
        for i, r in enumerate(roots):
            _print_node(r, "", i == len(roots) - 1)
        print("----------------------\n")



def create_checkpoint(dirpath: Union[str, Path], **kwargs) -> CheckpointManager:
    return CheckpointManager(dirpath=dirpath, **kwargs)
