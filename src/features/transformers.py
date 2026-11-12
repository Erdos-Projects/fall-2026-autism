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
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so src/utils.py can be imported
from utils import check_project_dir, parse_target_size, save_pt_settings, PROJECT_DIR

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
CLIP_PERCENTILE = 99.5                  # percentile limiting the brightness of the pixels
FG_PERCENTILE = (0.2, 99)               # foreground selection percentiles        
OVERWRITE = False                        # SET TO True TO REGENERATE .pt FILES THAT ALREADY EXIST
ZIP_PATH = Path(                        # CHANGE THIS TO WHERE YOUR ZIP FILE IS LOCATED                                      
    f"{PROJECT_DIR}/data/scans")        # note we assume the zip file is saved as "mri_{orientation}.zip"

INVENTORY_PATH = Path(f"{PROJECT_DIR}/data/data_inventory.csv")   


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def get_remove_set(data: pd.DataFrame) -> set[str]:
    '''
        get subject ids for those on stimulants at the time of the scan

        @param data : the data inventory

        @return remove_set : subject ids (as strings) to skip
    '''
    on_stims = 0
    remove_ids = data[data["OFF_STIMULANTS_AT_SCAN"] == on_stims]["SUB_ID"].to_numpy()
    return {str(sub_id) for sub_id in remove_ids}


# --------------------------------------------------------------------------- #
# Image Resize and Transform
# --------------------------------------------------------------------------- #
def nii_to_tensor(raw: bytes, name: str, target_shape: tuple[int]) -> torch.Tensor | None:
    '''
        normalize and resize one .nii scan

        @param raw : the bytes of the .nii file
        @param name : the scan name, used in log messages

        @return img_tensor : fp16 tensor of shape (1, *TARGET_SHAPE), or None if the foreground is nearly empty
    '''
    # loading the image
    img_nib = nib.Nifti1Image.from_bytes(raw)
    # put every scan in the same orientation so axes mean the same thing across sites
    img_nib = nib.as_closest_canonical(img_nib)
    # extract the img
    img_nib = np.squeeze(img_nib.get_fdata(dtype=np.float32))

    # per-scan intensity normalization BEFORE the fp16 cast. Raw intensities differ a lot
    # between scanners and can exceed the fp16 maximum (65504), which would turn into inf.
    # The mask is a crude foreground threshold (it includes skull/scalp), not a brain mask.
    foreground = img_nib > FG_PERCENTILE[0] * np.percentile(img_nib, FG_PERCENTILE[1])
    if foreground.sum() < 1000:
        print(f"skipping {name}: foreground mask is nearly empty")
        return None
    img_nib = np.clip(img_nib, 0, np.percentile(img_nib[foreground], CLIP_PERCENTILE))  # tame extreme bright voxels
    img_nib = (img_nib - img_nib[foreground].mean()) / (img_nib[foreground].std() + 1e-6)  # z-score
    img_nib[~foreground] = 0.0

    # unchanged: put the slice axis first.
    # With canonical orientation this gives axial slices: img[k] is slice k.
    img = img_nib.transpose(2, 0, 1)

    # convert the numpy array to tensor.
    img_tensor = torch.from_numpy(img)

    # resize to a fixed shape and add a channel dimension -> (1, D, H, W).
    # Conv3d expects this layout; for a 2D CNN over slices use img_tensor[0, k] as slice k.
    img_tensor = F.interpolate(img_tensor[None, None], size=target_shape,
                                mode="trilinear", align_corners=False)[0]

    # change to fp16 to reduce the size.
    # This step is not necessary and can be also removed,
    # but could help with training on machines with not alot of VRAM.
    return img_tensor.to(torch.float16)


# --------------------------------------------------------------------------- #
# File Extraction
# --------------------------------------------------------------------------- #
def extract_zip(zip_path: Path, data: pd.DataFrame, remove_set: set[str], orientation: str,
                target_shape: tuple[int]) -> None:
    '''
        extract and convert the MRI files in one zip to .pt files, skipping those belonging to removed subjects

        @param zip_path : path to the zip file; the .pt files are saved in a folder with the same name
        @param data : the data inventory
        @param remove_set : subject ids (as strings) to skip
        @param orientation : the orientation of the scan (either 'axial', 'sagittal', or 'coronal')
        @param target_shape : tuple of 3 integers to resize the scans to
    '''
    # diagnosis lookup so each saved tensor carries its label (ABIDE: 1 = autism, 2 = control)
    dx_by_sub = data.set_index("SUB_ID")["DX_GROUP"]

    save_dir = Path(f"{zip_path.with_suffix('')}/{'_'.join(map(str, target_shape))}")
    save_dir.mkdir(parents=True, exist_ok=True)
    n_skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.endswith(".nii"):
                continue
            if remove_set.intersection(name.split("_")):
                n_skipped += 1
                print(f"skipping {zip_path.name}:{member.filename}")
                continue

            # get save path for the tensor files
            save_path = save_dir / (Path(name).stem + ".pt")
            # remove empty *.pt folders left behind by an older version of this script
            if save_path.is_dir():
                save_path.rmdir()
            if save_path.exists() and not OVERWRITE:
                continue

            img_tensor = nii_to_tensor(zf.read(member), Path(name).stem, target_shape)
            if img_tensor is None:
                continue

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
    print(f"skipped {n_skipped} files from removed subjects in {zip_path}")
    print(f"Saving preprocessing settings in "
          f"{save_pt_settings(save_dir, orientation, target_shape, FG_PERCENTILE, CLIP_PERCENTILE)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--o", "--orientation", default="sagittal",
                    help="set scan orientation")
    ap.add_argument("--shape", default="(96,96,96)",
                    help="set image shape (format is (int, int, int))")
    args = ap.parse_args()

    target_shape = parse_target_size(args.shape)
    zip_file = Path(f"{ZIP_PATH}/mri_{args.o}.zip")
    
    check_project_dir(PROJECT_DIR)
    print("Using settings: ")
    print(f"  Zip files location: {ZIP_PATH}")
    print(f"  Shaping of images to: {target_shape}")
    print(f"  Will overwrite existing files: {OVERWRITE}")

    data = pd.read_csv(INVENTORY_PATH)
    remove_set = get_remove_set(data)

    # extract and convert MRI files in each zip in data/scans to pt files
    extract_zip(zip_file, data, remove_set, args.o, target_shape)


if __name__ == "__main__":
    main()
