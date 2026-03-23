# Advanced Usage Guide

This document dives into the core lifecycle of a Syckpt experiment, how to navigate its commit tree, manage branching, and utilize Mega-Hashes for pristine tracking.

## Branching & Run Modes

Syckpt defaults to `run_mode="new_branch"` for maximum safety—preventing you from accidentally overwriting hours of training because you restarted a script!

When initializing `CheckpointManager`, the `run_mode` controls how the current branch HEAD behaves:

1. **`new_branch`** (Default): If there is already a HEAD commit on the branch, Syckpt will automatically generate a new unique branch name (e.g. `exp_..._continue_...`) branching off the latest commit. This leaves your old branch untouched.
2. **`append`**: Resumes training from the HEAD of the current branch, continuously adding new commits linearly.
3. **`overwrite`**: Deletes the current branch entirely and starts a completely fresh, detached history on that branch name.

Example usage:
```python
# Starts a new branch if the default 'main' branch already has commits:
with CheckpointManager("./experiment") as ckpt:
    pass

# Forcefully purge the branch and start completely fresh:
with CheckpointManager("./experiment", run_mode="overwrite") as ckpt:
    pass
```

## Mega-Hashes (Session Squashing)

Often, a training loop runs for many epochs, generating dozens of commits. To prevent your commit tree from becoming a cluttered mess of linear hashes, Syckpt automatically "squashes" the commits from a single training session into a **Mega-Hash** upon exiting the context manager or completing a `ckpt.loop()`.

Mega-Hashes simply contain a list of `sub_commits` pointing to the actual epoch checkpoints. 
When navigating the history, the single Mega-Hash represents the entire training run, while the individual epoch checkpoints remain accessible through its sub-commit list!

```python
with CheckpointManager("./experiment") as ckpt:
    for epoch in ckpt.loop(epochs=10):
        # ... training log ...
        ckpt.save(metric=loss)

# Upon exit, you will see a single [MEGA-HASH] in your tree summarizing those 10 epochs.
```

## Navigating the Commit Tree

You can visually print the git-like tree representing your experiment's history:
```python
ckpt.print_tree()
```

To jump back in time to a specific epoch hash or branch to perform a hyperparameter sweep:
```python
# Exactly restores model, optimizer, scheduler, dataloader, and RNG to this commit!
ckpt.goto("hash_or_branch_name")
```

If you ever need to clean up an abandoned experiment:
```python
ckpt.delete_branch("exp_continue_1a2b")
```

## Best-K Dynamic Checkpointing

When passing a `metric` to `ckpt.save()`, Syckpt leverages lightweight Git-like Tags (`refs/tags/best_1`, `best_2`, etc.) to track your top performing checkpoints! 
To preserve delta-compression integrity, Syckpt does **not** physically delete sub-optimal checkpoints. Instead, it fluidly repoints the `best_X` tags towards the winning hashes.

```python
# Keeps track of the top 3 lowest validation losses
with CheckpointManager("./experiment", max_to_keep=3, maximize=False) as ckpt:
    # ...
    ckpt.save(metric=val_loss)
```
