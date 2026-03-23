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

def compute_delta(current: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Computes the difference between two flat tensor dictionaries, identifying frozen subsets."""
    delta = {}
    for k, v in current.items():
        if k in base and v.shape == base[k].shape and v.dtype == base[k].dtype:
            # Check if this entire layer is mathematically identical (e.g. frozen backbone)
            # torch.equal is extremely fast and prevents saving un-mutated blocks.
            if torch.equal(v, base[k]):
                delta[k] = {"__frozen__": k}
            else:
                # Execute difference on mutated layers for heavy gzip compressibility
                delta[k] = v - base[k]
        else:
            delta[k] = v
    return delta

def apply_delta(base: Dict[str, torch.Tensor], delta: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Applies a delta to a base flat tensor dictionary, resolving frozen links."""
    reconstructed = {}
    
    # We don't blind copy the whole base anymore to save massive RAM spikes
    for k, d in delta.items():
        if isinstance(d, dict) and "__frozen__" in d:
            # Virtual hard-link. This matrix didn't change at all! 
            # We reference the base instantly.
            reconstructed[k] = base[d["__frozen__"]].clone()
        elif k in base and torch.is_tensor(d) and d.shape == base[k].shape and d.dtype == base[k].dtype:
            # Delta Patch
            reconstructed[k] = base[k].clone() + d
        else:
            # Raw new tensor
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
        self.tags_dir = f"{self.syckpt_dir}/refs/tags"
        
        self.fs.makedirs(self.objects_dir, exist_ok=True)
        self.fs.makedirs(self.refs_dir, exist_ok=True)
        self.fs.makedirs(self.tags_dir, exist_ok=True)
        
        # Initialize HEAD
        head_path = f"{self.syckpt_dir}/HEAD"
        if not self.fs.exists(head_path):
            self.write_head("main")

    def _atomic_write_json(self, data: Any, path: str):
        class TensorEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, torch.Tensor):
                    return obj.item() if obj.numel() == 1 else obj.tolist()
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
                
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(data, tmp, cls=TensorEncoder)
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

    # Git-native tag management
    def write_tag(self, tag_name: str, commit_hash: str):
        """Sets a tag ref to a specific commit hash."""
        tag_path = f"{self.tags_dir}/{tag_name}"
        with self.fs.open(tag_path, "w") as f:
            f.write(commit_hash)

    def read_tag(self, tag_name: str) -> Optional[str]:
        """Reads the commit hash for a tag."""
        tag_path = f"{self.tags_dir}/{tag_name}"
        if not self.fs.exists(tag_path):
            return None
        with self.fs.open(tag_path, "r") as f:
            return f.read().strip()

    def list_tags(self) -> List[str]:
        """Lists all local tags."""
        if not self.fs.exists(self.tags_dir):
            return []
        tags = self.fs.ls(self.tags_dir, detail=False)
        return [t.split("/")[-1] for t in tags]

    def delete_tag(self, tag_name: str) -> bool:
        """Deletes a tag ref."""
        tag_path = f"{self.tags_dir}/{tag_name}"
        if self.fs.exists(tag_path):
            self.fs.rm(tag_path)
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

    def get_commit_tree(self) -> Dict[str, Any]:
        """Returns the commit graph by tracing back from all branch tips."""
        branches = self.list_branches()
        commits = {}
        branch_tips = {}
        
        for branch in branches:
            tip = self.read_ref(branch)
            if tip:
                branch_tips[branch] = tip
                curr = tip
                while curr and curr not in commits:
                    try:
                        c_data = self.load_commit(curr)
                        commits[curr] = c_data
                        
                        # Process sub-commits of MegaHashes so they appear in history!
                        if c_data.get("is_mega"):
                            for sub_hash in c_data.get("sub_commits", []):
                                try:
                                    sc_data = self.load_commit(sub_hash)
                                    commits[sub_hash] = sc_data
                                except FileNotFoundError:
                                    pass
                                    
                        curr = c_data.get("parent")
                    except FileNotFoundError:
                        break
                        
        # Append tags to output graph
        tags = {}
        for tag in self.list_tags():
            t_hash = self.read_tag(tag)
            if t_hash:
                tags[tag] = t_hash
                        
        return {"commits": commits, "branch_tips": branch_tips, "tags": tags}

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
        """Saves tensors, potentially using delta compression and layer-freezing if a base is provided."""
        blob_path = f"{self.objects_dir}/{blob_hash}.safetensors"
        
        metadata = {
            "blob_hash": blob_hash,
            "is_delta": False,
            "frozen_links": {}
        }
        
        if base_tensors is not None:
            # We have a base, compute delta map
            delta_map = compute_delta(tensors, base_tensors)
            
            # Safetensors CANNOT save dictionaries `{"__frozen__": "k"}`.
            # We must separate the pure floats from the virtual links.
            pure_tensors = {}
            for k, v in delta_map.items():
                if isinstance(v, dict) and "__frozen__" in v:
                    metadata["frozen_links"][k] = v["__frozen__"]
                else:
                    pure_tensors[k] = v
                    
            self._save_safetensors_fsspec(pure_tensors, blob_path)
            metadata["is_delta"] = True
            # base_hash tracking is handled by the caller commit metadata
        else:
            self._save_safetensors_fsspec(tensors, blob_path)
            
        return metadata
        
    def load_tensors(self, blob_hash: str, base_tensors: Optional[Dict[str, torch.Tensor]] = None, is_delta: bool = False, frozen_links: Optional[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
        """Loads tensors, resolving delta patches and hard-links if necessary."""
        blob_path = f"{self.objects_dir}/{blob_hash}.safetensors"
        if not self.fs.exists(blob_path):
            raise FileNotFoundError(f"Blob {blob_hash} not found in CAS storage.")
            
        loaded_tensors = self._load_safetensors_fsspec(blob_path)
        
        if is_delta:
            if base_tensors is None:
                raise ValueError(f"Blob {blob_hash} is a delta, but no base_tensors were provided.")
            
            # Re-inject the frozen string pointers so apply_delta can process them natively
            delta_map = dict(loaded_tensors)
            if frozen_links:
                for k, frozen_key in frozen_links.items():
                    delta_map[k] = {"__frozen__": frozen_key}
                    
            return apply_delta(base_tensors, delta_map)
        else:
            return loaded_tensors
