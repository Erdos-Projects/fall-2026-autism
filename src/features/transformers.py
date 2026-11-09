'''
    Extract the scans from their zip files for all subjects except for those on stimulants at the time of the scan
    Resize the scans according to the TARGET_SHAPE variable and save them as .pt files 
    Get scan id, subject id, and label (DX) from the file name to save in the tensor

    Authors: Doyee Byun, Jess Miears
'''
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import nibabel as nib
import torch
import torch.nn.functional as F
import os

proj_dir = os.environ.get("PROJECT_DIR")

# settings for the extraction and conversion of MRI files
ZIP_PATH = Path(f"{proj_dir}/data/scans")   # CHANGE THIS TO WHERE YOUR ZIP FILES ARE LOCATED
TARGET_SHAPE = (96, 96, 96)                 # CHANGE THIS TO THE IMAGE SIZE YOU WANT TO RESIZE TO
OVERWRITE = False                           # SET TO True TO REGENERATE .pt FILES THAT ALREADY EXIST
print("Using settings: ")
print(f"  Zip files location: {ZIP_PATH}")
print(f"  Shaping of images to: {TARGET_SHAPE}")
print(f"  Will overwrite existing files: {OVERWRITE}")


# file paths
scans_path = Path(f"{proj_dir}/data/scans")
inventory_path = Path(f"{proj_dir}/data/data_inventory.csv")
data = pd.read_csv(inventory_path)

# get subject ids for those on stimulants 
on_stims = 0
remove_ids = data[data["OFF_STIMULANTS_AT_SCAN"] == on_stims]["SUB_ID"].to_numpy()
remove_set = {str(sub_id) for sub_id in remove_ids}

# extract and convert MRI files in each zip in data/scans to pt files
# skipping those belonging to removed subjects
for ZIP_PATH in scans_path.glob("*.zip"):
    save_dir = ZIP_PATH.with_suffix("")
    save_dir.mkdir(exist_ok=True)
    n_skipped = 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.endswith(".nii"):
                continue
            if remove_set.intersection(name.split("_")):
                n_skipped += 1
                print(f"skipping {ZIP_PATH.name}:{member.filename}")
                continue

            # get save path for the tensor files
            save_path = save_dir / (Path(name).stem + ".pt")
            # remove empty *.pt folders left behind by an older version of this script
            if save_path.is_dir():
                save_path.rmdir()
            if save_path.exists() and not OVERWRITE:
                continue

            # loading the image
            img_nib = nib.Nifti1Image.from_bytes(zf.read(member))
            # put every scan in the same orientation so axes mean the same thing across sites
            img_nib = nib.as_closest_canonical(img_nib)
            # extract the img 
            img_nib = np.squeeze(img_nib.get_fdata(dtype=np.float32))
        
            # per-scan intensity normalization BEFORE the fp16 cast. Raw intensities differ a lot
            # between scanners and can exceed the fp16 maximum (65504), which would turn into inf.
            # The mask is a crude foreground threshold (it includes skull/scalp), not a brain mask.
            foreground = img_nib > 0.2 * np.percentile(img_nib, 99)
            if foreground.sum() < 1000:
                print(f"skipping {Path(name).stem}: foreground mask is nearly empty")
                continue
            img_nib = np.clip(img_nib, 0, np.percentile(img_nib[foreground], 99.5))  # tame extreme bright voxels
            img_nib = (img_nib - img_nib[foreground].mean()) / (img_nib[foreground].std() + 1e-6)  # z-score
            img_nib[~foreground] = 0.0
        
            # unchanged: put the slice axis first.
            # With canonical orientation this gives axial slices: img[k] is slice k.
            img = img_nib.transpose(2, 0, 1)
        
            # convert the numpy array to tensor.
            img_tensor = torch.from_numpy(img)
        
            # resize to a fixed shape and add a channel dimension -> (1, D, H, W).
            # Conv3d expects this layout; for a 2D CNN over slices use img_tensor[0, k] as slice k.
            img_tensor = F.interpolate(img_tensor[None, None], size=TARGET_SHAPE,
                                        mode="trilinear", align_corners=False)[0]
        
            # change to fp16 to reduce the size.
            # This step is not necessary and can be also removed,
            # but could help with training on machines with not alot of VRAM.
            img_tensor = img_tensor.to(torch.float16)
        
            # diagnosis lookup so each saved tensor carries its label (ABIDE: 1 = autism, 2 = control)
            dx_by_sub = data.set_index("SUB_ID")["DX_GROUP"]

            # parse scan id, subject id, and label (DX) from the file name to save in the tensor
            parts = Path(name).stem.split("_")
            if len(parts) < 2 or not parts[1].isdigit():
                print(f"skipping {Path(name).stem}: unexpected file name")
                continue
            scan_id, sub_id = parts[-1], int(parts[1])
            if str(sub_id) in remove_set:
                continue  # excluded subject (replaces deleting the file)
            if sub_id not in dx_by_sub.index:
                print(f"skipping {Path(name).stem}: subject {sub_id} not in inventory")
                continue
            label = int(dx_by_sub[sub_id] == 1)  # 1 = autism, 0 = control
        
            # saving the image
            torch.save({"vol": img_tensor, "id": sub_id, "label": label}, save_path)
    print(f"skipped {n_skipped} files from removed subjects in {ZIP_PATH}")