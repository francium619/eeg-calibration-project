"""
PyTorch port of pretrain.py: masked channel-window reconstruction across
the meta-training subjects' *calibration* sessions only (never touching the
held-out subjects, or the query/second-session data of the meta-training
subjects -- keeping the second session unseen during pretraining too means
the eventual cross-session eval in torch_eval_calibration.py isn't
contaminated at any stage of the pipeline).

Not executed in this sandbox -- see torch_model.py's docstring.
"""
import torch
import torch.nn.functional as F

from torch_model import EEGBackbone
from torch_data import MetaEEGDataLoader

SEED = 0
PRETRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7]
BATCH_SIZE = 32
N_STEPS = 3000
LR = 3e-3
MASK_FRACTION = 0.3
WIN_LEN = 50


def masked_reconstruction_loss(backbone, batch, mask_idx):
    _, tok = backbone.encode(batch, mask_idx=mask_idx)
    pred = backbone.decoder_head(tok)
    target = batch[:, mask_idx, :]
    return F.mse_loss(pred[:, mask_idx, :], target)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    pipeline = MetaEEGDataLoader(subject_ids=PRETRAIN_SUBJECTS)
    n_windows = pipeline.n_timepoints // WIN_LEN

    # Pool only the *calibration* (first) session per subject -- see module docstring.
    pooled = []
    for sub_id in PRETRAIN_SUBJECTS:
        sessions = sorted(pipeline.subject_session_datasets[sub_id].keys())
        calib_ds = pipeline.subject_session_datasets[sub_id][sessions[0]]
        pooled.append(calib_ds.X)
    all_trials = torch.cat(pooled).to(device)  # (n_trials, C, T)

    backbone = EEGBackbone(n_channels=pipeline.n_channels, win_len=WIN_LEN, n_windows=n_windows).to(device)
    params = backbone.backbone_base_parameters() + list(backbone.decoder_head.base.parameters())
    opt = torch.optim.AdamW(params, lr=LR)

    n_mask = max(1, int(MASK_FRACTION * backbone.n_tokens))
    losses = []
    for step in range(N_STEPS):
        idx = torch.randint(0, all_trials.shape[0], (BATCH_SIZE,))
        batch = backbone.patchify(all_trials[idx])
        mask_idx = torch.randperm(backbone.n_tokens)[:n_mask]

        opt.zero_grad()
        loss = masked_reconstruction_loss(backbone, batch, mask_idx)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % 200 == 0 or step == N_STEPS - 1:
            print(f"  step {step:5d}/{N_STEPS}  masked-recon MSE (running avg) = {sum(losses[-200:]) / len(losses[-200:]):.4f}")

    torch.save(backbone.state_dict(), "backbone_pretrained.pt")
    print("Saved pretrained backbone to backbone_pretrained.pt")


if __name__ == "__main__":
    main()
