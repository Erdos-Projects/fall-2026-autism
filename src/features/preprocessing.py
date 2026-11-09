'''
    Main preprocessing pipeline that drops unused features, integrates engineered features, and can be reused in modeling.

    Author: Jess Miears
'''

import csv
import pandas as pd
from pathlib import Path
import torch
import scipy.ndimage as ndi
from scipy.signal import find_peaks
import numpy as np
import os
from os.path import join


# file locations
proj_dir = os.environ.get("PROJECT_DIR")
inventory_path = Path(f"{proj_dir}/data/data_inventory.csv")
csv_file = f"{proj_dir}/data/manifest.csv"

# remove incorrect values in patient data csv
bad_value = "-9999"
with open(inventory_path, newline="") as f:
    rows = list(csv.reader(f))
n_replaced = 0
for row in rows:
    for j, value in enumerate(row):
        if value.strip() == bad_value:
            row[j] = ""
            n_replaced += 1
with open(inventory_path, "w", newline="") as f:
    csv.writer(f, lineterminator="\n").writerows(rows)
print(f"Replaced {n_replaced} {bad_value} values with empty values in {inventory_path}.")

# get data inventory and split into control and autism groups
data = pd.read_csv(inventory_path)
control = data[data["DX_GROUP"] == 2]
autism = data[data["DX_GROUP"] == 1]


def get_scan_file(subj_id: int, names: list) -> str:
    '''
        find whether the scan file exists for the subject id

        @param subj_id : subject id
        @param names : list of files names to check for the scan file

        @return name : name of the scan file if it exists
    '''
    for name in names:
        if str(subj_id) in name.split("_"):
            return name  


def find_matter(file: str) -> dict[tuple]:
    ''' 
        look at the image file to find the white and grey matter indices

        @param file : image file name

        @return dictionary of tuples for (min, max) values for 'grey' and 'white' matter
    '''
    sigma = 1
    brain_vol = torch.load(file)
    brain_vol_data = brain_vol["vol"].float().numpy()
    filt = ndi.gaussian_filter(brain_vol_data, sigma=sigma)
    filt_hist = ndi.histogram(filt, min=0, max=np.max(filt), bins=50)
    gr = np.gradient(filt_hist)
    idx1, _ = find_peaks(gr, distance=1)
    idx2, _ = find_peaks(-gr, distance=1)
    idx = np.sort(np.concatenate([idx1[:4], idx2[:4]]))
    if len(idx) < 5:
        return {'grey' : None, 'white' : None}
    return {'grey' : (int(idx[1]), int(idx[2]-1)), 'white' : (int(idx[2]), int(idx[4]))}


def process_files(scan_type: str) -> dict:
    '''
        loop through subject IDs to process the .pt files

        @param scan_type : the orientation of the scan (either 'axial', 'sagittal', or 'coronal')

        @return dictionary of ids and their grey matter and white matter tuples from find_matter
    '''
    print(f"Processing {scan_type} files. This may take a while...")
    scan_dir = Path(f"{proj_dir}/data/scans/mri_{scan_type}")
    all_ids = data["SUB_ID"].unique()
    names = [p.name for p in scan_dir.iterdir() if p.is_file() and p.suffix == ".pt"]
    grey = []
    white = []
    for id in all_ids:
        file = get_scan_file(id, names)
        if file is not None:
            matter = find_matter(join(scan_dir, file))
            grey.append(matter["grey"])
            white.append(matter["white"])
        else:
            grey.append(None)
            white.append(None)
    return {"grey" : grey, "white" : white}


def get_pt_info(subj_id: int, scan_type: str) -> dict:
    '''
        get the path to the .pt file for a given subject id and scan type

        @param subj_id : subject id
        @param scan_type : the orientation of the scan (either 'axial', 'sagittal', or 'coronal')

        @return path : path to the .pt file if it exists, else None
    '''
    scan_dir = Path(f"{proj_dir}/data/scans/mri_{scan_type}")
    names = [p.name for p in scan_dir.iterdir() if p.is_file() and p.suffix == ".pt"]
    file = get_scan_file(subj_id, names)
    if file is not None:
        return {"path": str(join(scan_dir, file)), "scan_id": Path(file).stem.split("_")[-1]}
    return {"path": None, "scan_id": None}


# get file paths
ax_paths = [get_pt_info(subj_id, "axial") for subj_id in data["SUB_ID"]]
sag_paths = [get_pt_info(subj_id, "sagittal") for subj_id in data["SUB_ID"]]
cor_paths = [get_pt_info(subj_id, "coronal") for subj_id in data["SUB_ID"]]

# get scan types for each subject
ax_matter = process_files("axial")
sag_matter = process_files("sagittal")
cor_matter = process_files("coronal")

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
print(f"Saving to CSV file {csv_file}")
manifest = pd.DataFrame(
    {
        "sub_id": data["SUB_ID"],
        "site": data["SITE_ID"],
        "label": data["DX_GROUP"].map({1: 1, 2: 0}),  # 1 = autism, 0 = control
        "age": data["AGE_AT_SCAN"],
        "sex": data["SEX"],
        "aq_outlier": data["SUB_ID"].isin(aq_outliers["SUB_ID"]),
        "unclear_diag": data["SUB_ID"].isin(unclear_diag["SUB_ID"]),

        "axial_scan_id": [p["scan_id"] for p in ax_paths],
        "axial_pt_path": [p["path"] for p in ax_paths],
        "axial_grey": ax_matter["grey"],
        "axial_white": ax_matter["white"],  

        "sagittal_scan_id": [p["scan_id"] for p in sag_paths],
        "sagittal_pt_path": [p["path"] for p in sag_paths],
        "sagittal_grey": sag_matter["grey"],
        "sagittal_white": sag_matter["white"],
        
        "coronal_scan_id": [p["scan_id"] for p in cor_paths],
        "coronal_pt_path": [p["path"] for p in cor_paths],
        "coronal_grey": cor_matter["grey"],
        "coronal_white": cor_matter["white"]
    }
)
manifest.to_csv(csv_file, index=False)
