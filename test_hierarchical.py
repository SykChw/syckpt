import torch
import shutil
import os
from syckpt import CheckpointManager

# Clean any existing experiment data
if os.path.exists("./test_hierarchical"):
    shutil.rmtree("./test_hierarchical")

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

with CheckpointManager("./test_hierarchical", run_mode="new_branch", max_to_keep=100) as ckpt:
    ckpt.model = model
    ckpt.optimizer = optimizer
    
    # Level 1 mega-hash (wraps 3 runs)
    for run in ckpt.loop(3, message="Experiment Runs"):
        # Level 0 mega-hash (wraps 5 epochs each)
        for ep in ckpt.loop(5, message=f"Run {run}"):
            
            # Pretend to train
            loss = 1.0 / (ep + 1)
            ckpt.save(metric=loss, message=f"ep-{ep}")

print("\n\n--- Verification Output ---")
tree = ckpt.storage.get_commit_tree()
print(f"Total commits in tree: {len(tree['commits'])}")
