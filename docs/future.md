# Future Outlook: Hierarchical Mega-Hashes

As training scales from local runs on single nodes to massive foundational model experiments distributed across clusters spanning months, tracking every single checkpoint becomes a UI and storage navigation nightmare.

While Syckpt currently squashes continuous "Session Commits" into **Mega-Hashes** (representing single contiguous blocks of runtime), the true evolution of this tree structure is **Hierarchical Mega-Hashes**.

## The Vision

Imagine nesting Mega-Hashes inside larger Mega-Hashes!

1. **Epoch Mega-Hashes**: Squashes 10,000 highly granular gradient step-checkpoints into a single `Epoch-1` commit.
2. **Phase Mega-Hashes**: Squashes the 100 `Epoch-X` commits into a `Warmup-Phase` or `Cooldown-Phase` commit.
3. **Experiment Mega-Hashes**: Combines multiple parallel phases and finetuning forks into a single `Run-V2` representation.

The graph output would look something like this:
```
--- Syckpt Tree ---
└── mega_experiment_v2 (HEAD, *main*) [MEGA-HASH]
    ├── mega_warmup [MEGA-HASH]
    │   ├── mega_epoch_1 [MEGA-HASH] 
    │   │   ├── 1a2b3c4d [Step 100]
    │   │   ├── ...
    ...
```

Instead of collapsing just lists of commits, Hierarchical Mega-Hashes will contain Directed Acyclic Graphs (DAGs) of sub-mega-hashes! This provides a massive zoom-in/zoom-out capability for researching very deep training histories without ever losing granular step rewindability!

### Why isn't it implemented yet?
To build this safely, the tree rendering algorithm and commit traversal endpoints need an interactive visualization engine beyond standard terminal output. 
Additionally, recursive delta-compression resolution across nested trees requires a highly optimized lazy tensor fetching strategy to prevent cascading I/O bottlenecks when seeking granular sub-commits embedded deep within multi-layer Mega-Hashes.

This is on our immediate roadmap for `v2.0`!
