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

# delete MRI files belonging to removed subjects
# filenames look like I328734_50340_MP-RAGE.nii, so match the subject id as a whole "_"-separated token
remove_set = {str(sub_id) for sub_id in remove_ids}
i=0
for path in Path("data/MRI").rglob("*.nii"):
    if path.is_file() and remove_set.intersection(path.name.split("_")):
        i+=1
        if i > len(remove_ids):
            print("WARNING: More files removed than IDs in remove_ids. Stopping file removal.")
            quit()
        print(f"removing {path}")
        path.unlink()

# save all the nii images as pt files
for file_path in Path("data/MRI").rglob("*.nii"):
    save_path = os.path.join(Path(file_path).resolve().parent, "pt_files/")
    # loading the image
    img_nib = nib.load(file_path)
    # extract the img.
    img_nib = img_nib.get_fdata()
    img = img_nib.transpose(2, 0, 1)

    # convert the numpy array to tensor.
    img_tensor = torch.from_numpy(img)

    # change to fp16 to reduce the size. 
    # This step is not necessary and can be also removed, 
    # but could help with training on machines with not alot of VRAM.
    img_tensor = img_tensor.to(torch.float16)

    # create the path and name for the file to save.
    path = os.path.join(save_path , file_path.stem  + ".pt")
    Path(save_path).mkdir(parents=False, exist_ok=True)

    # saving the image 
    torch.save({"vol": img_tensor, "id": 1}, path)