import os
import torch
import torch.nn as nn
from tempfile import TemporaryDirectory
from syckpt.manager import CheckpointManager

def test_sublayer_freezing_hardlinks():
    with TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        
        # 1. Build a dummy backbone and classification head
        model = nn.Sequential(
            nn.Linear(100, 100), # Backbone
            nn.ReLU(),
            nn.Linear(100, 10)   # Head
        )
        manager.register(model=model)
        
        # Commit 1
        hash1 = manager.save(message="Base initialization")
        
        # Block until async write completes
        import time
        time.sleep(1.0) 
        
        # 2. Freeze the backbone structurally!
        # In a real training script, the SGD step simply wouldn't alter the math block.
        # We simulate this cleanly by only altering the final head layer weights explicitly.
        with torch.no_grad():
            model[2].weight.add_(0.1)
            
        # Commit 2 (Delta Patch)
        hash2 = manager.save(message="Mutated Head only")
        time.sleep(1.0)
        
        # 3. Mathematically evaluate the backend JSON storage reference maps
        commit1 = manager.storage.load_commit(hash1)
        commit2 = manager.storage.load_commit(hash2)
        
        assert commit2["blob_metadata"]["is_delta"] == True
        
        # 4. Verify the __frozen__ pointer array mapping
        frozen_metadata = commit2["blob_metadata"].get("frozen_links", {})
        
        # model.0.weight should be mathematically identical 
        assert "model.0.weight" in frozen_metadata
        assert "model.0.bias" in frozen_metadata
        
        # model.2.weight was mutated, and thus should NOT be a zero-cost linked reference
        assert "model.2.weight" not in frozen_metadata
        
        # 5. Verify the actual reloaded weights are universally pure
        reloaded_model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)   
        )
        manager.load_into_model(reloaded_model, hash2)
        
        assert torch.equal(reloaded_model[0].weight, model[0].weight), "Backbone linked resolution failed"

if __name__ == "__main__":
    test_sublayer_freezing_hardlinks()
    print("test_sublayer_freezing_hardlinks PASSED.")
