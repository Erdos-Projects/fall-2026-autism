'''
    Extract the scans from their zip files for all subjects except for those on stimulants at the time of the scan
'''
import zipfile
import pandas as pd
from pathlib import Path
import nibabel as nib
import torch


# CHANGE THIS TO WHERE YOUR ZIP FILES ARE LOCATED
zip_path = Path("data/scans")

# dont change anything else
scans_path = Path("data/scans")
inventory_path = Path("data/data_inventory.csv")
data = pd.read_csv(inventory_path)

# get subject ids for those on stimulants 
on_stims = 0
remove_ids = data[data["OFF_STIMULANTS_AT_SCAN"] == on_stims]["SUB_ID"].to_numpy()
remove_set = {str(sub_id) for sub_id in remove_ids}

# extract and convert MRI files in each zip in data/scans to pt files
# skipping those belonging to removed subjects
for zip_path in scans_path.glob("*.zip"):
    save_dir = zip_path.with_suffix("")
    save_dir.mkdir(exist_ok=True)
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
            save_path = save_dir / (Path(name).stem + ".pt")
            if save_path.exists():
                continue

            # loading the image
            img_nib = nib.Nifti1Image.from_bytes(zf.read(member))
            # extract the img.
            img = img_nib.get_fdata().transpose(2, 0, 1)

            # convert the numpy array to tensor
            img_tensor = torch.from_numpy(img)

            # change to fp16 to reduce the size
            img_tensor = img_tensor.to(torch.float16)

            # saving the image
            torch.save({"vol": img_tensor, "id": 1}, save_path)
            print(f"saved {save_path}")
    print(f"skipped {n_skipped} files from removed subjects in {zip_path}")