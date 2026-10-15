'''
    Main preprocessing pipeline that drops unused features, integrates engineered features, and can be reused in modeling.
'''

import csv
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import nibabel as nib
import torch

scans_path = Path("data/scans")

'''
    remove incorrect values in patient data csv
'''
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


'''
    remove subjects based on criteria
'''
# values we need to remove
aq_cutoff = 19
on_stims = 0

# get the subject id we are removing from the dataset
remove_ids = data[(data["DX_GROUP"] == 1) & (data['AQ_TOTAL'] < aq_cutoff)]["SUB_ID"].to_numpy()
remove_ids = np.append(
    remove_ids, data[(data["DX_GROUP"] == 2) 
                     & (data['AQ_TOTAL'] > aq_cutoff)]["SUB_ID"].to_numpy()
    )
remove_ids = np.append(remove_ids, data[data["OFF_STIMULANTS_AT_SCAN"] == on_stims]["SUB_ID"].to_numpy())


'''
    convert MRI files in each zip in data/scans to pt files, skipping those belonging to removed subjects
'''
# zip members look like ABIDE/50383/.../ABIDE_50383_MRI_MP-RAGE_..._I328282.nii, so match the subject id
# as a whole "_"-separated token of the filename
# the zips are left untouched; each .nii is read straight from its zip (never written to disk) and saved
# as a pt file in the folder with the same name as its zip, e.g. mri_axial.zip -> mri_axial/
remove_set = {str(sub_id) for sub_id in remove_ids}
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

