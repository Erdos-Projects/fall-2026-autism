'''
    Main preprocessing pipeline that drops unused features, integrates engineered features, and can be reused in modeling.
'''

import csv
import pandas as pd
import numpy as np
from pathlib import Path

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
for path in Path("data/MRI").rglob("*"):
    if path.is_file() and remove_set.intersection(path.name.split("_")):
        i+=1
        if i > len(remove_ids):
            print("WARNING: More files removed than IDs in remove_ids. Stopping file removal.")
            quit()
        print(f"removing {path}")
        path.unlink()