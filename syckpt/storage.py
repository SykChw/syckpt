import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import fsspec
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)

def flatten_state(state: Any, prefix: str = "", tensors: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Any, Dict[str, torch.Tensor]]:
    """Separates a nested Python dictionary into a structure mapping and a flat dictionary of tensors."""
    if tensors is None:
        tensors = {}
        
    if isinstance(state, torch.Tensor):
        tensors[prefix] = state
        return {"__tensor__": prefix}, tensors
    elif isinstance(state, dict):
        result = {}
        for k, v in state.items():
            sub_prefix = f"{prefix}.{k}" if prefix else str(k)
            result[k], _ = flatten_state(v, sub_prefix, tensors)
        return result, tensors
    elif isinstance(state, list):
        result = []
        for i, v in enumerate(state):
            sub_prefix = f"{prefix}[{i}]"
            res, _ = flatten_state(v, sub_prefix, tensors)
            result.append(res)
        return result, tensors
    elif isinstance(state, tuple):
        result = []
        for i, v in enumerate(state):
            sub_prefix = f"{prefix}[{i}]"
            res, _ = flatten_state(v, sub_prefix, tensors)
            result.append(res)
        return {"__tuple__": result}, tensors
    else:
        # Primitives: str, int, float, bool, None
        return state, tensors

def unflatten_state(structure: Any, tensors: Dict[str, torch.Tensor]) -> Any:
    """Reconstructs a nested Python dictionary from a structure mapping and a flat dictionary of tensors."""
    if isinstance(structure, dict):
        if "__tensor__" in structure:
            return tensors[structure["__tensor__"]]
        elif "__tuple__" in structure:
            return tuple(unflatten_state(v, tensors) for v in structure["__tuple__"])
        else:
            return {k: unflatten_state(v, tensors) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [unflatten_state(v, tensors) for v in structure]
    else:
        return structure

def compute_delta(current: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Computes the difference between two flat tensor dictionaries."""
    delta = {}
    for k, v in current.items():
        if k in base and v.shape == base[k].shape and v.dtype == base[k].dtype:
            # We can only perform subtraction on same shape and dtype
            delta[k] = v - base[k]
        else:
            delta[k] = v
    return delta

def apply_delta(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Applies a delta to a base flat tensor dictionary."""
    reconstructed = {}
    # First copy all base tensors
    for k, v in base.items():
        reconstructed[k] = v.clone()
        
    for k, d in delta.items():
        if k in reconstructed and d.shape == reconstructed[k].shape and d.dtype == reconstructed[k].dtype:
            reconstructed[k] = reconstructed[k] + d
        else:
            reconstructed[k] = d
    return reconstructed

class CASStorage:
    """Content-Addressable Storage using fsspec and safetensors with delta compression and Git-like abstractions."""
    
    def __init__(self, root: str):
        self.root = root
        self.fs, self.fs_path = fsspec.core.url_to_fs(root)
        self.syckpt_dir = f"{self.fs_path}/.syckpt"
        self.objects_dir = f"{self.syckpt_dir}/objects"
        self.refs_dir = f"{self.syckpt_dir}/refs/heads"
        
        self.fs.makedirs(self.objects_dir, exist_ok=True)
        self.fs.makedirs(self.refs_dir, exist_ok=True)
        
        # Initialize HEAD
        head_path = f"{self.syckpt_dir}/HEAD"
        if not self.fs.exists(head_path):
            self.write_head("main")

    def _atomic_write_json(self, data: Any, path: str):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name
        try:
            self.fs.put_file(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _read_json(self, path: str) -> Any:
        with self.fs.open(path, "r") as f:
            return json.load(f)

    # Git-native refs management
    def write_head(self, branch_name: str):
        """Sets the HEAD symbolic ref to a branch."""
        head_path = f"{self.syckpt_dir}/HEAD"
        with self.fs.open(head_path, "w") as f:
            f.write(f"ref: refs/heads/{branch_name}")

    def read_head(self) -> str:
        """Reads HEAD and returns the current branch name."""
        head_path = f"{self.syckpt_dir}/HEAD"
        if not self.fs.exists(head_path):
            return "main"
        with self.fs.open(head_path, "r") as f:
            content = f.read().strip()
            if content.startswith("ref: refs/heads/"):
                return content.split("refs/heads/")[-1]
            return "main"

    def write_ref(self, branch_name: str, commit_hash: str):
        """Sets a branch ref to a specific commit hash."""
        ref_path = f"{self.refs_dir}/{branch_name}"
        with self.fs.open(ref_path, "w") as f:
            f.write(commit_hash)

    def read_ref(self, branch_name: str) -> Optional[str]:
        """Reads the commit hash for a branch."""
        ref_path = f"{self.refs_dir}/{branch_name}"
        if not self.fs.exists(ref_path):
            return None
        with self.fs.open(ref_path, "r") as f:
            return f.read().strip()

    def list_branches(self) -> List[str]:
        """Lists all local branches."""
        if not self.fs.exists(self.refs_dir):
            return []
        refs = self.fs.ls(self.refs_dir, detail=False)
        return [r.split("/")[-1] for r in refs]

    def delete_ref(self, branch_name: str) -> bool:
        """Deletes a branch ref."""
        ref_path = f"{self.refs_dir}/{branch_name}"
        if self.fs.exists(ref_path):
            self.fs.rm(ref_path)
            return True
        return False

    # Git-native object management
    def save_commit(self, commit_hash: str, commit_data: Dict[str, Any]):
        """Saves a commit metadata object (Tree+Commit equivalent)."""
        commit_path = f"{self.objects_dir}/{commit_hash}.json"
        self._atomic_write_json(commit_data, commit_path)

    def load_commit(self, commit_hash: str) -> Dict[str, Any]:
        """Loads a commit metadata object."""
        commit_path = f"{self.objects_dir}/{commit_hash}.json"
        if not self.fs.exists(commit_path):
            raise FileNotFoundError(f"Commit {commit_hash} not found.")
        return self._read_json(commit_path)
        
    def check_commit_exists(self, commit_hash: str) -> bool:
        return self.fs.exists(f"{self.objects_dir}/{commit_hash}.json")

    def _save_safetensors_fsspec(self, tensors: Dict[str, torch.Tensor], path: str):
        """Saves a safetensors file over fsspec using a local temporary file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
            tmp_path = tmp.name
        try:
            # Safetensors requires a string path, so we save to local tmp first
            save_file(tensors, tmp_path)
            # Use fsspec to move to destination (could be S3, local, etc)
            self.fs.put_file(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    def _load_safetensors_fsspec(self, path: str) -> Dict[str, torch.Tensor]:
        """Loads a safetensors file over fsspec using a local temporary file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
            tmp_path = tmp.name
        try:
            self.fs.get_file(path, tmp_path)
            # Safetensors loads standard dict
            return load_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def save_tensors(self, tensors: Dict[str, torch.Tensor], blob_hash: str, base_tensors: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Saves tensors, potentially using delta compression if a base is provided."""
        blob_path = f"{self.objects_dir}/{blob_hash}.safetensors"
        
        metadata = {
            "blob_hash": blob_hash,
            "is_delta": False,
        }
        
        if base_tensors is not None:
            # We have a base, compute delta
            delta_tensors = compute_delta(tensors, base_tensors)
            self._save_safetensors_fsspec(delta_tensors, blob_path)
            metadata["is_delta"] = True
            # base_hash tracking is handled by the caller commit metadata
        else:
            self._save_safetensors_fsspec(tensors, blob_path)
            
        return metadata
        
    def load_tensors(self, blob_hash: str, base_tensors: Optional[Dict[str, torch.Tensor]] = None, is_delta: bool = False) -> Dict[str, torch.Tensor]:
        """Loads tensors, resolving delta if necessary."""
        blob_path = f"{self.objects_dir}/{blob_hash}.safetensors"
        if not self.fs.exists(blob_path):
            raise FileNotFoundError(f"Blob {blob_hash} not found in CAS storage.")
            
        loaded_tensors = self._load_safetensors_fsspec(blob_path)
        
        if is_delta:
            if base_tensors is None:
                raise ValueError(f"Blob {blob_hash} is a delta, but no base_tensors were provided.")
            return apply_delta(base_tensors, loaded_tensors)
        else:
            return loaded_tensors
