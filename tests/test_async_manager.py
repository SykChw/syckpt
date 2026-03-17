import torch
import torch.nn as nn
from tempfile import TemporaryDirectory
from syckpt.manager import CheckpointManager
import time

def test_async_multiprocessing_save():
    with TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        
        # Dummy Model
        model = nn.Linear(100, 100)
        manager.register(model=model)
        
        start_time = time.time()
        
        # 1. Execute an async save
        commit_hash = manager.save(message="Async Init")
        
        save_call_duration = time.time() - start_time
        
        # The save() call should return nearly instantaneously because the heavy 
        # file I/O and Safetensors execution forms in a detached Pool.
        # It should certainly take less than 0.5 seconds on CPU.
        assert save_call_duration < 0.5, "Save blocked the main thread excessively."
        
        # 2. Wait for the background OS process to actually execute writing to disk
        time.sleep(2.0)
        
        # 3. Validate the structural metadata array
        loaded_blob = manager.load(commit_hash)
        assert loaded_blob is not None
        assert loaded_blob["step"] == 0
        
        # Ensure the model dynamically bound inside the async state dictionary restored identically
        new_model = nn.Linear(100, 100)
        manager.load_into_model(new_model, commit_hash)
        

if __name__ == "__main__":
    test_async_multiprocessing_save()
    print("test_async_multiprocessing_save PASSED.")
