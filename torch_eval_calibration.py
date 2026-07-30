"""
PyTorch port of eval_calibration.py, on real, held-out subjects (not used in
meta-training) with a real cross-session split: calibrate on that subject's
first-recorded session, evaluate on their second session (a different day).
This is the realistic version of the NumPy prototype's headline comparison:
zero-shot vs. random-init-LoRA vs. meta-learned-LoRA, at matched calibration
budgets, on data the meta-learner never saw.

Not executed in this sandbox (see torch_model.py docstring). Before
trusting numbers from this script: check that `zero_shot` accuracy is well
above the 1/4 chance rate for 4-class motor imagery, and that the
meta-learned curve is at or above the random-init curve at every step count
-- if either of those doesn't hold, revisit backbone pretraining quality or
meta-training hyperparameters before concluding anything about LoRA vs.
meta-LoRA specifically (the NumPy prototype hit exactly this failure mode
early on: a poorly-pretrained backbone made *every* calibration strategy
look equally bad, which is a representation problem, not a calibration
-strategy problem -- see README's honest-limitations section).
"""
import numpy as np
import torch
import torch.nn.functional as F

from torch_model import EEGBackbone, ClassifierHead, load_lora_state_dict
from torch_data import MetaEEGDataLoader, EVENTS

EVAL_SUBJECTS = [8, 9]           # held out of meta-training's [1..7]
CALIB_SIZE = 12
STEP_CHECKPOINTS = [0, 1, 2, 3, 5, 8, 12]
LR = 0.08
GRAD_CLIP = 5.0
N_REPEATS = 4


def accuracy(backbone, head, X, y, device):
    backbone.eval(); head.eval()
    with torch.no_grad():
        patches = backbone.patchify(X.to(device))  # (B, C, T) raw trial -> (B, tokens, win_len)
        _, tok = backbone.encode(patches)
        preds = head(tok).argmax(-1).cpu()
    return float((preds == y).float().mean())


def fine_tune_curve(backbone, head, params, X_calib, y_calib, X_query, y_query, steps_ckpt, lr, device):
    opt = torch.optim.SGD(params, lr=lr)  # fresh, momentum-free -- see torch_meta_train.py docstring
    X_calib, y_calib = X_calib.to(device), y_calib.to(device)
    calib_patches = backbone.patchify(X_calib)
    accs, done = [], 0
    for target in steps_ckpt:
        while done < target:
            backbone.train(); head.train()
            opt.zero_grad()
            _, tok = backbone.encode(calib_patches)
            logits = head(tok)
            loss = F.cross_entropy(logits, y_calib)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            done += 1
        accs.append(accuracy(backbone, head, X_query, y_query, device))
    return accs


def fresh_backbone_and_head(device, n_channels, win_len, n_windows, pretrained_path="backbone_pretrained.pt"):
    backbone = EEGBackbone(n_channels=n_channels, win_len=win_len, n_windows=n_windows).to(device)
    try:
        backbone.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    except FileNotFoundError:
        print(f"WARNING: {pretrained_path} not found -- using an UNTRAINED backbone. "
              f"Run the pretraining step first for a meaningful result.")
    backbone.freeze_backbone_base()
    head = ClassifierHead(n_classes=len(EVENTS)).to(device)
    return backbone, head


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = MetaEEGDataLoader(subject_ids=EVAL_SUBJECTS)

    meta_state = torch.load("meta_lora_init.pt", map_location=device)

    results = {"zero_shot": [], "random_lora": [], "meta_lora": []}
    for sub_id in EVAL_SUBJECTS:
        for rep in range(N_REPEATS):
            calib_loader, query_loader = pipeline.get_subject_session_split_loaders(
                sub_id, calib_size=CALIB_SIZE, batch_size=CALIB_SIZE, seed=1000 * sub_id + rep
            )
            X_calib, y_calib = next(iter(calib_loader))
            X_query_all, y_query_all = zip(*[(x, y) for x, y in query_loader])
            X_query, y_query = torch.cat(X_query_all), torch.cat(y_query_all)

            win_len, n_windows = 50, pipeline.n_timepoints // 50

            # (a) zero-shot: meta-learned init, no adaptation
            bb, head = fresh_backbone_and_head(device, pipeline.n_channels, win_len, n_windows)
            load_lora_state_dict(bb, meta_state["lora"])
            head.load_state_dict(meta_state["head"])
            results["zero_shot"].append(accuracy(bb, head, X_query, y_query, device))

            # (b) random-init LoRA, fine-tuned from scratch
            bb_r, head_r = fresh_backbone_and_head(device, pipeline.n_channels, win_len, n_windows)
            params_r = bb_r.lora_parameters() + head_r.parameters_for_calibration()
            curve_r = fine_tune_curve(bb_r, head_r, params_r, X_calib, y_calib, X_query, y_query, STEP_CHECKPOINTS, LR, device)
            results["random_lora"].append(curve_r)

            # (c) meta-learned LoRA init, fine-tuned
            bb_m, head_m = fresh_backbone_and_head(device, pipeline.n_channels, win_len, n_windows)
            load_lora_state_dict(bb_m, meta_state["lora"])
            head_m.load_state_dict(meta_state["head"])
            params_m = bb_m.lora_parameters() + head_m.parameters_for_calibration()
            curve_m = fine_tune_curve(bb_m, head_m, params_m, X_calib, y_calib, X_query, y_query, STEP_CHECKPOINTS, LR, device)
            results["meta_lora"].append(curve_m)

        print(f"  subject {sub_id} done")

    zero_shot_mean = np.mean(results["zero_shot"])
    random_curve = np.mean(results["random_lora"], axis=0)
    meta_curve = np.mean(results["meta_lora"], axis=0)

    print(f"\nZero-shot (meta init, no calibration): {zero_shot_mean:.3f}  (chance = {1/len(EVENTS):.3f})")
    print(f"{'steps':>6} | {'random-init LoRA':>18} | {'meta-learned LoRA':>18}")
    for s, r, m in zip(STEP_CHECKPOINTS, random_curve, meta_curve):
        print(f"{s:6d} | {r:18.3f} | {m:18.3f}")


if __name__ == "__main__":
    main()
