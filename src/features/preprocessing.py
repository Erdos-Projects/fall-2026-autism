'''
    Main preprocessing pipeline that drops unused features, integrates engineered features, and can be reused in modeling.

    Author: Jess Miears
'''

import csv
import pandas as pd
from pathlib import Path
import os
from os.path import join
import argparse

ap = argparse.ArgumentParser(
    description="Make manifest ABIDE data for autism vs. control classification.")
ap.add_argument("--orientation", "-o", type=str, default="sagittal", choices=["axial", "sagittal", "coronal"],
                help="which scan orientation to use (default: axial)")
args = ap.parse_args()

# file locations
PROJECT_DIR = os.environ.get("PROJECT_DIR")
INVENTORY_PATH = Path(f"{PROJECT_DIR}/data/data_inventory.csv")
CLEAN_INVENTORY_PATH = Path(f"{PROJECT_DIR}/data/data_inventory_cleaned.csv")
CSV_FILE = f"{PROJECT_DIR}/data/manifest.csv"
ORIENTATION = args.orientation
print(f"Using {ORIENTATION} to build manifest.")

# remove incorrect values in patient data csv
bad_value = "-9999"
with open(INVENTORY_PATH, newline="") as f:
    rows = list(csv.reader(f))
n_replaced = 0
for row in rows:
    for j, value in enumerate(row):
        if value.strip() == bad_value:
            row[j] = ""
            n_replaced += 1
with open(CLEAN_INVENTORY_PATH, "w", newline="") as f:
    csv.writer(f, lineterminator="\n").writerows(rows)
print(f"Replaced {n_replaced} {bad_value} values with empty values in {INVENTORY_PATH}.")

# get data inventory and split into control and autism groups
data = pd.read_csv(CLEAN_INVENTORY_PATH)
control = data[data["DX_GROUP"] == 2]
autism = data[data["DX_GROUP"] == 1]


# UDFs
def get_scan_file(subj_id: int, names: list) -> str | None:
    '''
        find whether the scan file exists for the subject id

        @param subj_id : subject id
        @param names : list of files names to check for the scan file

        @return name : name of the scan file if it exists
    '''
    for name in names:
        if str(subj_id) in name.split("_"):
            return name  
    return None


def get_pt_info(subj_id: int, scan_type: str) -> dict:
    '''
        get the path to the .pt file for a given subject id and scan type

        @param subj_id : subject id
        @param scan_type : the orientation of the scan (either 'axial', 'sagittal', or 'coronal')

        @return path : path to the .pt file if it exists, else None
    '''
    scan_dir = Path(f"{PROJECT_DIR}/data/scans/mri_{scan_type}")
    names = [p.name for p in scan_dir.iterdir() if p.is_file() and p.suffix == ".pt"]
    file = get_scan_file(subj_id, names)
    if file:
        return {"path": str(join(scan_dir, file)), "scan_id": Path(file).stem.split("_")[-1]}
    return {"path": None, "scan_id": None}


# get file paths
scan_info = [get_pt_info(subj_id, ORIENTATION) for subj_id in data["SUB_ID"]]

# get aq outliers
aq_border = 19
aq_outliers = pd.concat([
    control[control["AQ_TOTAL"] >= aq_border],
    autism[autism["AQ_TOTAL"] <= aq_border],
])
print("Marked AQ_TOTAL outliers.")

# get subjects with unclear diagnoses
unclear_diag = data[(data["DSM_IV_TR"] >= 3)]
print("Marked those with an unclear diagnosis.")

# save df of all important information for each subject
print(f"Saving to CSV file {CSV_FILE}")
manifest = pd.DataFrame(
    {
        "sub_id": data["SUB_ID"],
        "site": data["SITE_ID"],
        "label": data["DX_GROUP"].map({1: 1, 2: 0}),  # 1 = autism, 0 = control
        "age": data["AGE_AT_SCAN"],
        "sex": data["SEX"],
        "aq_outlier": data["SUB_ID"].isin(aq_outliers["SUB_ID"]),
        "unclear_diag": data["SUB_ID"].isin(unclear_diag["SUB_ID"]),
        "scan_id": [p["scan_id"] for p in scan_info],
        "pt_path": [p["path"] for p in scan_info]
    }
)

# remove subjects with no scans
manifest = manifest.dropna(
    subset=["scan_id"], how="all"
)  

# save to csv
manifest.to_csv(CSV_FILE, index=False)
