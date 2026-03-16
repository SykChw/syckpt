"""Main CheckpointManager - Git-like checkpoint system with LSH hashing and CAS storage."""

import os
import json
import fcntl
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from checkpoint.config import HyperConfig
from checkpoint.hash import LSHHashGenerator
from checkpoint.state import (
    StateManager,
    get_rng_state,
    set_rng_state,
    get_deterministic_state,
    set_deterministic_state,
)
from checkpoint.storage import CASStorage, flatten_state, unflatten_state

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
        "blob_metadata"
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
        blob_metadata: Optional[Dict] = None
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
            "blob_metadata": self.blob_metadata
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
            data.get("blob_metadata", {})
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
    ):
        self.root = str(dirpath)
        self.storage = CASStorage(self.root)
        
        self.max_to_keep = max_to_keep
        self.maximize = maximize
        self.auto_resume = auto_resume
        self.save_rng = save_rng

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

        # Load latest commit from the active branch if it exists
        branch_hash = self.storage.read_ref(self._current_branch)
        if branch_hash:
            self._hash = branch_hash
            self._load_commit_into_cache(branch_hash)

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
            "state_manager"
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
        blob_metadata = commit_data.get("blob_metadata", {})
        blob_hash = blob_metadata.get("blob_hash", commit_hash)
        
        if blob_metadata.get("is_delta"):
            base_hash = commit_data.get("parent")
            if not base_hash:
                raise ValueError(f"Commit {commit_hash} requires delta resolution but has no parent.")
            base_tensors = self._fetch_tensors(base_hash)
            return self.storage.load_tensors(blob_hash, base_tensors=base_tensors, is_delta=True)
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
            metadata, flat_tensors = self._build_commit_data()
            
            if self.save_rng:
                metadata["rng"] = rng_state

            # Optional delta compression
            base_tensors = None
            if base_hash and self.storage.check_commit_exists(base_hash):
                base_tensors = self._fetch_tensors(base_hash)

            blob_hash = current_hash # using same naming for blob
            blob_metadata = self.storage.save_tensors(flat_tensors, blob_hash, base_tensors=base_tensors)

            commit_data = {
                "hash": current_hash,
                "parent": base_hash,
                "message": message or "Update",
                "metric": metric,
                "blob_hash": blob_metadata["blob_hash"],
                "blob_metadata": blob_metadata,
                **metadata
            }

            self.storage.save_commit(current_hash, commit_data)
            self.storage.write_ref(self._current_branch, current_hash)
            self.storage.write_head(self._current_branch)
            self._hash = current_hash
            self._commits[current_hash] = Commit.from_dict(commit_data)
            
            if is_dist:
                dist.barrier()
                
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
        for ep in range(start, epochs):
            self._epoch = ep
            if steps_per_epoch is None:
                yield ep
            else:
                for st in range(steps_per_epoch):
                    self._step = start * steps_per_epoch + st
                    yield ep, st
            if self.auto_resume:
                self.save()

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

        if self.auto_resume:
            latest = self.storage.read_ref(self._current_branch)
            if latest and self.storage.check_commit_exists(latest):
                try:
                    self.load(latest)
                    logger.info(
                        f"Resumed from step {self._step}, batch {self._batch_idx}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to resume from {latest}: {e}")
            else:
                # Need initial save
                self._hash = self._generate_hash()
                logger.info(f"Initialized new branch {self._current_branch} with {self._hash}")
        else:
            self._hash = self._generate_hash()

        return self

    def __exit__(self, *args):
        try:
            self.save()
        except Exception as e:
            logger.warning(f"Failed to save on exit: {e}")
        finally:
            self._lock_release()
        return False


def create_checkpoint(dirpath: Union[str, Path], **kwargs) -> CheckpointManager:
    return CheckpointManager(dirpath=dirpath, **kwargs)
