import nbformat
import sys
import subprocess

with open("autoencoder.ipynb", "r") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code" and "for epoch in ckpt.loop" in cell.source:
        # We rewrite the cell source to use a two-level hierarchy
        new_source = """\
# ─── Training loop with hierarchical syckpt ────────────────────────────────────

syckpt_dir = os.path.join(EXPERIMENT, ".syckpt")
if FRESH_START and os.path.exists(syckpt_dir):
    shutil.rmtree(syckpt_dir)
    print(f"Cleared existing checkpoint dir: {syckpt_dir}")

history = {"total": [], "recon": [], "kl": []}

with CheckpointManager(EXPERIMENT, max_to_keep=5, run_mode="new_branch") as ckpt:
    ckpt.model     = model
    ckpt.optimizer = optimizer
    ckpt.scheduler = scheduler
    ckpt.sampler   = sampler
    ckpt.config    = {"input_dim": INPUT_DIM, "latent_dim": LATENT_DIM, "lr": LR}

    # Outer loop: Level 1 Mega-Hash
    for phase in ckpt.loop(3, message="Training Phases"):
        print(f"\\n--- Starting Phase {phase+1}/3 ---")
        
        # Inner loop: Level 0 Mega-Hash
        for epoch in ckpt.loop(epochs=EPOCHS//3, message=f"Phase {phase+1} Epochs"):
            model.train()
            # linearly anneal beta from 0 → BETA over KL_ANNEAL epochs,
            # then hold at BETA — prevents KL from being crushed at the start
            beta_t = min(BETA, BETA * (epoch + 1) / KL_ANNEAL)
            total_loss = recon_loss = kl_loss = 0.0
            n_batches  = 0

            for (batch_x,) in dataloader:
                batch_x = batch_x.to(device)

                x_hat, mu_z, log_var_z = model(batch_x)
                loss, recon, kl = vae_loss(batch_x, x_hat, mu_z, log_var_z, beta=beta_t, free_bits=FREE_BITS)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                recon_loss += recon.item()
                kl_loss    += kl.item()
                n_batches  += 1
                ckpt.step_up()

            scheduler.step()

            avg_loss  = total_loss / n_batches
            avg_recon = recon_loss / n_batches
            avg_kl    = kl_loss    / n_batches

            history["total"].append(avg_loss)
            history["recon"].append(avg_recon)
            history["kl"].append(avg_kl)

            print(
                f"Phase {phase+1} Epoch {epoch:>3}/{EPOCHS//3}  |  "
                f"Loss: {avg_loss:.5f}  |  Recon: {avg_recon:.5f}  |  KL: {avg_kl:.5f}  |  "
                f"beta: {beta_t:.3f}"
            )
            # Save leaf commit
            ckpt.save(metric=avg_loss, message=f"phase-{phase+1}_ep-{epoch}")
"""
        cell.source = new_source

with open("autoencoder.ipynb", "w") as f:
    nbformat.write(nb, f)

# Execute the notebook to ensure it works and to generate the output
subprocess.run([
    "/home/sykchw/miniconda3/bin/jupyter", "nbconvert", 
    "--to", "notebook", 
    "--execute", 
    "--inplace", 
    "autoencoder.ipynb"
], check=True)
