"""
Baseline 3D CNN for ABIDE T1 scans (autism vs. control).

Input : data/manifest.csv written by preprocess_patched.py
        (columns: scan_id, sub_id, label, pt_path, site, age, sex)
Output: best_model.pt            weights from the epoch with the best validation AUC
        data/manifest_split.csv  the manifest plus a train/val/test column (saved so every
                                 later experiment can reuse exactly the same split)

How to run:
    python train_baseline.py --smoke        # tiny, fast run to check that everything works
    python train_baseline.py                # the real baseline run

Key ideas, in plain words:
  * The model reads a (1, 96, 96, 96) volume and outputs ONE number (a "logit").
    sigmoid(logit) is the model's estimated probability that the subject has autism.
  * Training = show the model batches of scans, compare its guess with the true label
    (the "loss"), and nudge the weights to reduce the loss. One pass over all training
    scans is an "epoch".
  * After each epoch we measure AUC on the VALIDATION set (subjects the model never trains
    on) and keep the weights that score best. The TEST set is only touched once, at the end.
  * The split is by SUBJECT, so all scans of one person are in the same set.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
MANIFEST_IN = Path("data/manifest.csv")
MANIFEST_SPLIT = Path("data/manifest_split.csv")
CHECKPOINT = Path("best_model.pt")

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MIN_STRATUM_SIZE = 6      # site/diagnosis groups smaller than this are merged when splitting

LEARNING_RATE = 3e-4      # how big each weight update is; too high = unstable, too low = slow
WEIGHT_DECAY = 1e-2       # mild penalty on large weights; helps against overfitting
PATIENCE = 8              # stop if validation AUC hasn't improved for this many epochs


def set_seed(seed):
    """Make runs repeatable (up to small GPU nondeterminism)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 1. Subject-level split
# --------------------------------------------------------------------------- #
def make_split(manifest):
    """
    Add a 'split' column. We split SUBJECTS (not scans), stratified on diagnosis and site so
    each set has a similar mix. If a site/diagnosis group is too small to stratify, we merge
    small groups, and if that still fails we stratify on diagnosis only.
    """
    subj = manifest.drop_duplicates("sub_id")[["sub_id", "site", "label"]].reset_index(drop=True)

    strata = subj["label"].astype(str) + "_" + subj["site"].astype(str)
    sizes = strata.map(strata.value_counts())
    strata = strata.where(sizes >= MIN_STRATUM_SIZE, subj["label"].astype(str) + "_rare")

    def stratified(idx, test_size):
        for col in (strata, subj["label"]):                 # try site+label, then label only
            try:
                return train_test_split(idx, test_size=test_size,
                                        stratify=col.iloc[idx], random_state=SEED)
            except ValueError:
                print("  stratification failed; trying a coarser grouping")
        raise ValueError("could not stratify the split")

    idx = np.arange(len(subj))
    trainval_idx, test_idx = stratified(idx, TEST_FRAC)
    train_idx, val_idx = stratified(trainval_idx, VAL_FRAC / (1 - TEST_FRAC))

    subj["split"] = "train"
    subj.loc[val_idx, "split"] = "val"
    subj.loc[test_idx, "split"] = "test"

    out = manifest.merge(subj[["sub_id", "split"]], on="sub_id", how="left")
    assert out.groupby("sub_id")["split"].nunique().max() == 1, "a subject ended up in two splits"
    return out


# --------------------------------------------------------------------------- #
# 2. Data loading
# --------------------------------------------------------------------------- #
def augment(vol):
    """
    Light augmentation, applied to training scans only. Each epoch the model sees slightly
    different versions of the same scan, which makes memorizing individual scans harder.
      - random shift of up to 4 voxels along each axis
      - random brightness scale and a little noise
    (No left-right flipping: brain asymmetry might matter, so we don't assume it doesn't.)
    """
    shifts = tuple(int(s) for s in torch.randint(-4, 5, (3,)))
    vol = torch.roll(vol, shifts=shifts, dims=(1, 2, 3))
    vol = vol * (0.9 + 0.2 * torch.rand(1)) + 0.05 * torch.randn_like(vol)
    return vol


class ScanDataset(Dataset):
    """Loads one preprocessed (1, D, H, W) tensor per row of the manifest."""

    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        vol = torch.load(row["pt_path"])["vol"].float()     # stored as fp16, compute in fp32
        if self.train:
            vol = augment(vol)
        label = torch.tensor(float(row["label"]))
        return vol, label, i                                 # i lets us map predictions back to rows


# --------------------------------------------------------------------------- #
# 3. The model
# --------------------------------------------------------------------------- #
def conv_block(c_in, c_out):
    """3x3x3 convolution -> batch norm -> ReLU -> halve the size with max pooling."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, padding=1),    # looks for local 3D patterns
        nn.BatchNorm3d(c_out),                               # keeps activations well-scaled
        nn.ReLU(inplace=True),                               # non-linearity
        nn.MaxPool3d(2),                                     # 96 -> 48 -> 24 -> 12 -> 6
    )


class Baseline3DCNN(nn.Module):
    """
    Four conv blocks (channels 8 -> 16 -> 32 -> 64), then global average pooling (collapses
    each of the 64 feature maps to one number, so the head doesn't depend on the input size),
    dropout, and one linear layer that outputs a single logit.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 8), conv_block(8, 16), conv_block(16, 32), conv_block(32, 64))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.pool(self.features(x))).squeeze(1)   # shape (batch,)


# --------------------------------------------------------------------------- #
# 4. Evaluation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, df, batch_size, device):
    """Return df with a 'prob' column (model's probability of autism for each scan)."""
    loader = DataLoader(ScanDataset(df), batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()                                             # turns off dropout, fixes batch norm
    probs = np.zeros(len(df))
    for vol, _, idx in loader:
        probs[idx.numpy()] = torch.sigmoid(model(vol.to(device))).cpu().numpy()
    out = df.reset_index(drop=True).copy()
    out["prob"] = probs
    return out


def per_subject(pred_df):
    """Average the scan-level probabilities so every subject counts once."""
    return pred_df.groupby("sub_id").agg(
        label=("label", "first"), site=("site", "first"), prob=("prob", "mean")).reset_index()


def safe_auc(labels, probs):
    """AUC is undefined if only one class is present; return NaN in that case."""
    return roc_auc_score(labels, probs) if len(set(labels)) == 2 else float("nan")


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline works")
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- data and split ----
    manifest = pd.read_csv(MANIFEST_IN)
    if MANIFEST_SPLIT.exists():
        manifest = pd.read_csv(MANIFEST_SPLIT)               # reuse the saved split
    else:
        manifest = make_split(manifest)
        manifest.to_csv(MANIFEST_SPLIT, index=False)
        print(f"saved split to {MANIFEST_SPLIT}")

    if args.smoke:                                           # keep a handful of scans per split
        manifest = manifest.groupby("split", group_keys=False).head(8)
        args.epochs = 2

    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(f"{name}: {len(d)} scans, {d['sub_id'].nunique()} subjects, "
              f"autism fraction {d.drop_duplicates('sub_id')['label'].mean():.2f}")

    train_loader = DataLoader(ScanDataset(train_df, train=True), batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))

    # ---- model, loss, optimizer ----
    model = Baseline3DCNN().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # If one class is more common, weight the rarer class up so the model doesn't just
    # predict the majority class.
    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)    # loss for a yes/no label
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ---- training loop with early stopping on validation AUC ----
    best_auc, epochs_without_gain = -1.0, 0
    history = []                                             # one row per epoch, saved at the end
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for vol, label, _ in train_loader:
            vol, label = vol.to(device), label.to(device)
            optimizer.zero_grad()                            # clear old gradients
            loss = loss_fn(model(vol), label)                # how wrong was the model?
            loss.backward()                                  # work out how to change each weight
            optimizer.step()                                 # change the weights a little
            running_loss += loss.item() * len(label)
        train_loss = running_loss / len(train_df)

        val_subj = per_subject(predict(model, val_df, args.batch_size, device))
        val_auc = safe_auc(val_subj["label"], val_subj["prob"])
        score = 0.5 if np.isnan(val_auc) else val_auc
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val AUC {val_auc:.3f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

        if score > best_auc:
            best_auc, epochs_without_gain = score, 0
            torch.save(model.state_dict(), CHECKPOINT)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                print(f"no improvement for {PATIENCE} epochs; stopping early")
                break

    pd.DataFrame(history).to_csv("training_log.csv", index=False)   # used by the inspection notebook

    # ---- final test evaluation (done once, with the best-validation weights) ----
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    test_subj = per_subject(predict(model, test_df, args.batch_size, device))
    preds = (test_subj["prob"] >= 0.5).astype(int)
    print("\n=== TEST RESULTS (subject level) ===")
    print(f"subjects: {len(test_subj)}")
    print(f"AUC     : {safe_auc(test_subj['label'], test_subj['prob']):.3f}")
    print(f"accuracy: {accuracy_score(test_subj['label'], preds):.3f}")
    print(f"majority-class accuracy (what a model that ignores the scan would get): "
          f"{max(test_subj['label'].mean(), 1 - test_subj['label'].mean()):.3f}")

    print("\nPer-site results (small sites give very noisy numbers):")
    for site, g in test_subj.groupby("site"):
        site_preds = (g["prob"] >= 0.5).astype(int)
        print(f"  {site:12s} n={len(g):3d}  AUC={safe_auc(g['label'], g['prob']):.3f}  "
              f"acc={accuracy_score(g['label'], site_preds):.3f}")


if __name__ == "__main__":
    main()
