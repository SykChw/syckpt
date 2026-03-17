import torch
from torch.utils.data import Sampler, DataLoader
from typing import Iterator, Optional, Sized, List

class StatefulRandomSampler(Sampler[int]):
    """
    A Sampler that yields random indices but explicitly maintains state allowing
    for exact $O(1)$ index list resumption upon process crashes.
    """
    def __init__(self, data_source: Sized, batch_size: int, base_seed: int = 42):
        self.data_source = data_source
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.epoch = 0
        self.batch_idx = 0
        
        self._indices: List[int] = []
        self._generator = torch.Generator()
        
    def __iter__(self) -> Iterator[int]:
        # Seed explicitly based on epoch to ensure deterministic layout
        self._generator.manual_seed(self.base_seed + self.epoch)
        n = len(self.data_source)
        
        # Generate the complete unshuffled array for the epoch deterministically
        self._indices = torch.randperm(n, generator=self._generator).tolist()
        
        # O(1) mathematical fast-forward slice!
        # If we crashed at batch 500, we instantly drop the first (500 * batch_size) indices!
        items_to_skip = self.batch_idx * self.batch_size
        
        if items_to_skip > 0:
            yield from self._indices[items_to_skip:]
        else:
            yield from self._indices
            
    def __len__(self) -> int:
        return len(self.data_source) - (self.batch_idx * self.batch_size)
        
    def state_dict(self):
        return {
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "base_seed": self.base_seed,
            "indices_length_cache": len(self._indices)
        }
        
    def load_state_dict(self, state):
        self.epoch = state.get("epoch", 0)
        self.batch_idx = state.get("batch_idx", 0)
        self.base_seed = state.get("base_seed", 42)
