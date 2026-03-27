# Advanced Usage Guide

This document covers the full lifecycle of a `syckpt` experiment: run modes, Mega-Hash commit squashing, tree navigation, hyperparameter sweeps, and Best-K checkpointing.

---

## Run Modes

Every `CheckpointManager` takes a `run_mode` argument that controls what happens when you re-enter a context manager on a directory that already has commits.

| `run_mode` | Behaviour |
|---|---|
| `"new_branch"` **(default)** | Fork a **new branch** every run. Each run is isolated and independently rewindable. |
| `"append"` | Resume from the current branch tip and continue adding commits linearly. |
| `"overwrite"` | Delete the current branch's history entirely and start fresh. |

### `new_branch` (default)

```python
# Run 1 → creates branch: main_continue_a1b2
with CheckpointManager("./experiment") as ckpt:
    ckpt.model = model
    for epoch in ckpt.loop(epochs=50):
        ckpt.save(metric=val_loss)

# Run 2 → creates branch: main_continue_c3d4 (independent fork)
with CheckpointManager("./experiment") as ckpt:
    ckpt.model = model
    for epoch in ckpt.loop(epochs=50):
        ckpt.save(metric=val_loss)
```

Each run forks its own branch, initialized with the model weights from the **last best checkpoint** of the previous run. Epoch and step counters **reset to 0** — you always train for the full number of epochs you specify.

> **Why load weights from the previous run?** This lets sequential experiments benefit from transfer learning — each new run starts warm, not from scratch. If you want a true cold start, use `run_mode="overwrite"`.

### `append`

Continues the current branch from where it left off. Use this to extend training without creating a new branch:

```python
# First run: epochs 0–49
with CheckpointManager("./experiment", run_mode="append") as ckpt:
    ckpt.model = model
    for epoch in ckpt.loop(epochs=50):
        ckpt.save(metric=val_loss)

# Second run: resumes from epoch 49, trains epochs 50–99
with CheckpointManager("./experiment", run_mode="append") as ckpt:
    ckpt.model = model
    for epoch in ckpt.loop(epochs=100):
        ckpt.save(metric=val_loss)
```

### `overwrite`

Wipes the current branch tip and starts completely fresh on the same branch name. Useful for resetting a failed experiment.

```python
with CheckpointManager("./experiment", run_mode="overwrite") as ckpt:
    ckpt.model = model
    for epoch in ckpt.loop(epochs=50):
        ckpt.save(metric=val_loss)
```

---

## Mega-Hashes

When a training loop finishes, `syckpt` **squashes** all the individual epoch commits from that session into a single **Mega-Hash** commit. This keeps your tree clean — instead of 50 root-level hashes for 50 epochs, you see one Mega-Hash node with 50 sub-commits nested inside it.

```
--- Syckpt Tree ---
├── mega_9b2 (main_continue_3d32): [MEGA-HASH] 50 sub-commits | Loop Mega-Hash (50 epochs) [Epoch 49]
│   ├── 9af15b33: epoch-0 [Epoch 0]
│   ├── 9af15b33-24c3f9: epoch-1 [Epoch 1]
│   ├── ...
│   └── 9af15b33-e390dc: epoch-49 [Epoch 49]
└── mega_56c (HEAD, *main_continue_64e2*): [MEGA-HASH] 50 sub-commits | Loop Mega-Hash (50 epochs) [Epoch 49]
    ├── ...
```

### How does the Mega-Hash identifier work?

Each epoch save uses **Locality-Sensitive Hashing (LSH)** to generate a hash from your model architecture and hyperparameters. Because LSH is locality-sensitive, similar configurations produce *similar* hashes — all hashes within a single training run share the same 8-character LSH prefix (e.g. `9af15b33`).

When weights diverge across epochs (due to gradient updates), the hash suffix changes — you get collision-resolved variants like `9af15b33-24c3f9`, `9af15b33-a3b1c2`. This means the shared prefix **already encodes the experiment's identity**: every sub-commit from the same training configuration clusters under the same LSH root.

The Mega-Hash is a UUID-prefixed container (`mega_<uuid8>`) that groups these related sub-commits, making the tree readable at a glance while preserving full granularity underneath.

### Manually grouping commits

If you save commits without `ckpt.loop()` (manual epoch loop), you can explicitly group them:

```python
with CheckpointManager("./experiment") as ckpt:
    ckpt.model = model
    for epoch in range(50):
        ckpt._epoch = epoch
        # ... training ...
        ckpt.save(metric=val_loss)
# group_commits is called automatically in __exit__
```

Or call it yourself for mid-run squashing:

```python
ckpt.group_commits(message="Phase 1 warmup (epochs 0-10)")
```

---

## Navigating the Commit Tree

### Print the tree

```python
ckpt.print_tree()
```

Output shows all branches, mega-hashes, sub-commits, HEAD, and Best-K tags in a single tree view.

### Jump to any commit or branch

```python
# Restore model + optimizer + RNG to an exact epoch checkpoint
ckpt.goto("9af15b33-24c3f9")

# Switch to another branch by name
ckpt.goto("main_continue_3d32")
```

`goto()` resolves both branch names and raw commit hashes. For Mega-Hash commits, it automatically resolves to the last real sub-commit.

### Load() transparently resolves Mega-Hashes

```python
# This loads the last sub-commit's actual weights, not the mega-hash metadata
ckpt.load("mega_9b2c4d1e")
```

### View history of current branch

```python
for commit in ckpt.log(n=10):
    print(commit.hash, commit.epoch, commit.metric)
```

### Compare hyperparams between two commits

```python
diff = ckpt.diff("9af15b33", "8bde2c41")
print(diff["config_diff"])  # {'lr': {'v1': 0.001, 'v2': 0.0003}}
```

### Delete a branch

```python
ckpt.delete_branch("main_continue_a1b2")  # Cannot delete "main"
```

---

## Hyperparameter Sweeps

`new_branch` mode is purpose-built for hyperparameter sweeps. Each run forks an independent branch, so you can compare exactly where different hyperparameters lead:

```python
import syckpt

for lr in [1e-3, 3e-4, 1e-4]:
    with CheckpointManager("./sweep", run_mode="new_branch", max_to_keep=3, maximize=False) as ckpt:
        ckpt.model = build_model()
        ckpt.optimizer = torch.optim.Adam(ckpt.model.parameters(), lr=lr)
        ckpt.config = {"lr": lr, "batch_size": 32}

        for epoch in ckpt.loop(epochs=50):
            val_loss = train_one_epoch(...)
            ckpt.save(metric=val_loss)

# After the sweep, print_tree shows one mega-hash per lr value.
# Best-K tags (best_1, best_2, best_3) point to the globally best checkpoints across all branches.
```

### Finding the best model after a sweep

```python
ckpt = CheckpointManager("./sweep", auto_resume=False)
tree = ckpt.storage.get_commit_tree()

# Best-K tags are globally maintained across all branches
for tag, commit_hash in tree["tags"].items():
    if tag.startswith("best_"):
        c = tree["commits"][commit_hash]
        print(f"{tag}: hash={commit_hash[:8]}, metric={c.get('metric'):.4f}, branch={c.get('branch')}")

# Load the single best checkpoint
ckpt.goto(tree["tags"]["best_1"])
```

Or use `diff()` to compare any two contenders:

```python
diff = ckpt.diff("9af15b33", "8bde2c41")
```

---

## Best-K Dynamic Checkpointing

When you pass `metric=` to `ckpt.save()`, `syckpt` maintains lightweight Git-style tags (`refs/tags/best_1`, `best_2`, …) pointing to the top-K best checkpoints seen across the **entire session**.

```python
# Track top 5 lowest validation losses (maximize=False means lower is better)
with CheckpointManager("./experiment", max_to_keep=5, maximize=False) as ckpt:
    for epoch in ckpt.loop(epochs=100):
        val_loss = evaluate()
        ckpt.save(metric=val_loss)
```

Tags are **repointed** as better checkpoints are found — no physical files are deleted, preserving the delta-compression chain. At any point:

```python
# Load the best checkpoint found so far
best_hash = ckpt.storage.read_tag("best_1")
ckpt.load(best_hash)
```

---

## Context Manager vs. Manual Usage

### With `ckpt.loop()` (recommended)

```python
with CheckpointManager("./experiment") as ckpt:
    ckpt.model = model
    for epoch in ckpt.loop(epochs=50):
        # ... training ...
        ckpt.save(metric=val_loss)
# ✅ Mega-hash auto-created, tree printed, workers joined before exit
```

### Without `ckpt.loop()` (manual)

```python
with CheckpointManager("./experiment") as ckpt:
    ckpt.model = model
    for epoch in range(50):
        ckpt._epoch = epoch
        ckpt.step_up()
        # ... training ...
        ckpt.save(metric=val_loss)
# ✅ Mega-hash still auto-created in __exit__ via group_commits
```

### Without context manager (e.g. Jupyter-style)

```python
ckpt = CheckpointManager("./experiment", run_mode="new_branch")
ckpt.model = model
ckpt.optimizer = optimizer

# Manual saves — no auto-grouping
for epoch in range(10):
    ckpt._epoch = epoch
    ckpt.save(metric=val_loss)

# Manually squash and print
ckpt.group_commits(message="My experiment")
ckpt.print_tree()
```

---

## Working with Branches Without the Context Manager

```python
ckpt = CheckpointManager("./experiment", auto_resume=False)

# Navigate to a specific branch
ckpt.goto("main_continue_a1b2")

# Fork from the current position
ckpt.create_branch("finetuning_run")

# Save from there
ckpt.save(message="initial finetune checkpoint")
```
