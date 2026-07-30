"""Renders eval_results.npz (produced by eval_calibration.py) as a PNG."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("eval_results.npz")
steps = d["step_checkpoints"]
random_curve, meta_curve = d["random_curve"], d["meta_curve"]
random_std, meta_std = d["random_std"], d["meta_std"]
zero_shot = d["zero_shot"]
n = d["random_all"].shape[0]

fig, ax = plt.subplots(figsize=(7.5, 5))

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (2-class)")
ax.axhline(np.mean(zero_shot), color="#888888", linestyle="--", linewidth=1.3,
           label=f"zero-shot (meta init, no calibration) = {np.mean(zero_shot):.3f}")

se_r = random_std / np.sqrt(n)
se_m = meta_std / np.sqrt(n)
ax.errorbar(steps, random_curve, yerr=se_r, marker="o", capsize=3,
            color="#d95f02", label="Random-init LoRA (fine-tuned from scratch)")
ax.errorbar(steps, meta_curve, yerr=se_m, marker="s", capsize=3,
            color="#1b9e77", label="Meta-learned LoRA init (fine-tuned)")

ax.set_xlabel("Calibration gradient steps on new subject (12 trials)")
ax.set_ylabel("Accuracy on held-out query trials\n(subjects unseen in pretraining & meta-training)")
ax.set_title("Few-shot subject calibration: meta-learned vs. random-init LoRA")
ax.set_ylim(0.40, 0.60)
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("calibration_comparison.png", dpi=150)
print("Saved calibration_comparison.png")
