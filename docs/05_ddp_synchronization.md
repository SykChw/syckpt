# 5. Distributed Data Parallel (DDP) Mechanics

When training massive parameter models spanning across 8, 32, or 1024 distinct GPUs on supercomputer clusters, coordinating file writes safely is a critical point of failure.

If all 8 GPUs attached to a node attempt to write gigabytes of overlapping checkpoints directly to `s3://` or the local disk simultaneously, race-conditions destroy the index files and randomly corrupt the blob payloads.

`syckpt` seamlessly integrates with native `torch.distributed` mechanisms.

## The Atomic `save()` DDP Implementation

When the training loop initiates `.save()`, the manager orchestration invokes strict hierarchy controls:

```python
    def save(self, metric: Optional[float] = None) -> str:
        
        # 1. Distributed Execution Halting
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            is_main = dist.get_rank() == 0
        else:
            is_main = True
```

### `dist.barrier()`

The first operation is locking. `dist.barrier()` forces any GPU that reaches `.save()` to halt execution immediately. It will freeze hardware progression infinitely until every single GPU rank remaining on the supercomputer node catches up and contacts the synchronization server.
This guarantees that the entire model is fully materialized across the NVLink topology before serialization attempts initiate.

### Isolating Rank 0 (Main Node)

Once the cluster is locked perfectly in sync, we assign write priorities. We flag `is_main = dist.get_rank() == 0`.
Only Rank 0 is granted physical permission to interact with the external hard drives or the `fsspec` network stream.

The remaining GPUs process standard state flattening locally, but they **sleep** when the physical `CASStorage` file creation routines activate. Rank 0 calculates the LSH hash dynamically, creates the `.json` Commit Metadata file, updates the Git `.syckpt/refs/heads/`, and pushes the `Delta-Compressed` float arrays.

### Safely Unlocking execution with `dist.broadcast_object_list`

The final obstacle involves desynchronization tracking. Because only Rank 0 executed the codebase dictating the Commit string identifier mappings in the Object Store, the secondary GPUs are now entirely blind to what commit `hash` structurally identifies their last layer.

```python
        # Ensure secondary nodes are aware of the LSH Hash 
        # that rank 0 mapped the cluster arrays over to prevent 
        # broken inheritance later
        if dist.is_available() and dist.is_initialized():
            hash_list = [self.hash] if is_main else [None]
            dist.broadcast_object_list(hash_list, src=0)
            self.hash = hash_list[0]
```

`syckpt` resolves this elegantly. GPU Rank 0 packages the string hash uniquely into an array vector length 1 `[self.hash]`.
It forces PyTorch to execute `dist.broadcast_object_list(...)` outward originating explicitly from `src=0` (Rank 0).

The serialized string packet fires instantaneously across the multi-GPU NVLink bridges, landing perfectly into the empty arrays of GPU 1 through N. All GPUs inject the updated local Hash vector, the barrier naturally exits bounds, and the training loop iterates forward completely synchronously without disk bottlenecks.
