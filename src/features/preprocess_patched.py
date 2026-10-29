'''
    Main preprocessing pipeline that drops unused features, integrates engineered features, and can be reused in modeling.
'''

import csv
import os
import pandas as pd
import numpy as np
from pathlib import Path
import nibabel as nib
import torch
import torch.nn.functional as F  # PATCH: needed to resize volumes

# PATCH: every volume is resized to this shape so scans from different sites can be batched
TARGET_SHAPE = (96, 96, 96)

# remove incorrect values in patient data csv
# -9999 is the missing value code, so replace it with an empty value
# done with the csv module (rather than pandas) so the rest of the file is written back unchanged
inventory_path = Path("data/data_inventory.csv")
missing_value = "-9999"
with open(inventory_path, newline="") as f:
    rows = list(csv.reader(f))
n_replaced = 0
for row in rows:
    for j, value in enumerate(row):
        if value.strip() == missing_value:
            row[j] = ""
            n_replaced += 1
with open(inventory_path, "w", newline="") as f:
    csv.writer(f, lineterminator="\n").writerows(rows)
print(f"replaced {n_replaced} {missing_value} values with empty values in {inventory_path}")

data = pd.read_csv(inventory_path)

# remove subjects based on criteria
# values we need to remove
aq_cutoff = 19
on_stims = 0

# get the subject id we are removing from the dataset
remove_ids = data[(data["DX_GROUP"] == 1) & (data['AQ_TOTAL'] < aq_cutoff)]["SUB_ID"].to_numpy()
remove_ids = np.append(remove_ids, data[(data["DX_GROUP"] == 2) & (data['AQ_TOTAL'] > aq_cutoff)]["SUB_ID"].to_numpy())
remove_ids = np.append(remove_ids, data[data["OFF_STIMULANTS_AT_SCAN"] == on_stims]["SUB_ID"].to_numpy())

# PATCH: report how many subjects the filters could not be applied to. Comparisons against
# missing (NaN) values are False, so these subjects are KEPT; it is worth knowing how many.
print(f"AQ_TOTAL missing for {data['AQ_TOTAL'].isna().sum()} of {len(data)} subjects (kept)")
print(f"OFF_STIMULANTS_AT_SCAN missing for {data['OFF_STIMULANTS_AT_SCAN'].isna().sum()} subjects (kept)")

# PATCH: instead of deleting the raw MRI files (unlink is permanent, so trying a different
# exclusion rule would mean re-downloading), remember the excluded subject ids and skip
# those scans in the conversion loop below. The old "i > len(remove_ids)" check and quit()
# are gone: a subject with several scans can legitimately have more files than ids.
# filenames look like I328734_50340_MP-RAGE.nii, so the subject id is the second "_"-separated token
remove_set = {str(sub_id) for sub_id in remove_ids}
print(f"{len(remove_set)} subjects excluded; their scans will be skipped (nothing is deleted)")

# PATCH: diagnosis lookup so each saved tensor carries its label (ABIDE: 1 = autism, 2 = control)
dx_by_sub = data.set_index("SUB_ID")["DX_GROUP"]

# save all the nii images as pt files
manifest_rows = []  # PATCH: collects one row per saved scan for the manifest written at the end
for file_path in list(Path("data/MRI").rglob("*.nii")):  # PATCH: list() so folders created below can't affect the walk
    # PATCH: parse scan id and subject id from the file name
    parts = file_path.name.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        print(f"skipping {file_path}: unexpected file name")
        continue
    scan_id, sub_id = parts[0], int(parts[1])
    if str(sub_id) in remove_set:
        continue  # excluded subject (replaces deleting the file)
    if sub_id not in dx_by_sub.index:
        print(f"skipping {file_path}: subject {sub_id} not in inventory")
        continue
    label = int(dx_by_sub[sub_id] == 1)  # 1 = autism, 0 = control

    save_path = os.path.join(Path(file_path).resolve().parent, "pt_files/")
    # loading the image
    img_nib = nib.load(file_path)
    # PATCH: put every scan in the same orientation so axes mean the same thing across sites
    img_nib = nib.as_closest_canonical(img_nib)
    # extract the img (PATCH: float32 instead of float64, squeeze drops a trailing dimension of 1)
    img_nib = np.squeeze(img_nib.get_fdata(dtype=np.float32))

    # PATCH: per-scan intensity normalization BEFORE the fp16 cast. Raw intensities differ a lot
    # between scanners and can exceed the fp16 maximum (65504), which would turn into inf.
    # The mask is a crude foreground threshold (it includes skull/scalp), not a brain mask.
    foreground = img_nib > 0.2 * np.percentile(img_nib, 99)
    if foreground.sum() < 1000:
        print(f"skipping {file_path}: foreground mask is nearly empty")
        continue
    img_nib = np.clip(img_nib, 0, np.percentile(img_nib[foreground], 99.5))  # tame extreme bright voxels
    img_nib = (img_nib - img_nib[foreground].mean()) / (img_nib[foreground].std() + 1e-6)  # z-score
    img_nib[~foreground] = 0.0

    # unchanged: put the slice axis first.
    # With canonical orientation this gives axial slices: img[k] is slice k.
    img = img_nib.transpose(2, 0, 1)

    # convert the numpy array to tensor.
    img_tensor = torch.from_numpy(img)

    # PATCH: resize to a fixed shape and add a channel dimension -> (1, D, H, W).
    # Conv3d expects this layout; for a 2D CNN over slices use img_tensor[0, k] as slice k.
    img_tensor = F.interpolate(img_tensor[None, None], size=TARGET_SHAPE,
                               mode="trilinear", align_corners=False)[0]

    # change to fp16 to reduce the size.
    # This step is not necessary and can be also removed,
    # but could help with training on machines with not alot of VRAM.
    img_tensor = img_tensor.to(torch.float16)

    # create the path and name for the file to save.
    path = os.path.join(save_path , file_path.stem  + ".pt")
    Path(save_path).mkdir(parents=False, exist_ok=True)

    # saving the image
    # PATCH: "id" is now the real subject id (it was 1 for every scan) and the label is stored too
    torch.save({"vol": img_tensor, "id": sub_id, "label": label}, path)
    manifest_rows.append({"scan_id": scan_id, "sub_id": sub_id, "label": label, "pt_path": path})

# PATCH: manifest with site / age / sex per scan, for the subject-level split and the
# demographic experiments later. Split on sub_id, never on individual scans.
manifest = pd.DataFrame(manifest_rows).merge(
    data[["SUB_ID", "SITE_ID", "AGE_AT_SCAN", "SEX"]].rename(
        columns={"SUB_ID": "sub_id", "SITE_ID": "site", "AGE_AT_SCAN": "age", "SEX": "sex"}),
    on="sub_id", how="left")
manifest.to_csv("data/manifest.csv", index=False)
print(f"saved {len(manifest)} scans from {manifest['sub_id'].nunique()} subjects; wrote data/manifest.csv")
