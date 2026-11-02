'''
    Main preprocessing pipeline that drops unused features, integrates engineered features, and can be reused in modeling.
'''

import csv
import pandas as pd
from pathlib import Path
import torch
import scipy.ndimage as ndi
from scipy.signal import find_peaks
import numpy as np
from os.path import join


unprocessed_data = pd.read_csv("data/data_inventory.csv")
control = unprocessed_data[unprocessed_data["DX_GROUP"] == 2]
autism = unprocessed_data[unprocessed_data["DX_GROUP"] == 1]


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
    scan_dir = Path(f"data/scans/mri_{scan_type}")
    all_ids = unprocessed_data["SUB_ID"].unique()
    names = [p.name for p in scan_dir.iterdir() if p.is_file() and p.suffix == ".pt"]
    ids = []
    grey = []
    white = []
    for id in all_ids:
        file = get_scan_file(id, names)
        if file is not None:
            matter = find_matter(join(scan_dir, file))
            ids.append(id)
            grey.append(matter["grey"])
            white.append(matter["white"])
    print(f"Processed {scan_type} files.")
    return {"id" : ids, "grey" : grey, "white" : white}


# remove incorrect values in patient data csv
inventory_path = Path("data/data_inventory.csv")
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

# now to save ids we want to flag for specific circumstances
# scan types for each subject
ax_ids = process_files("axial")
sag_ids = process_files("sagittal")
cor_ids = process_files("coronal")

# get aq outliers
aq_border = 19
aq_outliers = pd.concat([
    control[control["AQ_TOTAL"] >= aq_border],
    autism[autism["AQ_TOTAL"] <= aq_border],
])
print("Marked AQ_TOTAL outliers.")

# get subjects with unclear diagnoses
unclear_diag = unprocessed_data[(unprocessed_data["DSM_IV_TR"] >= 3)]
print("Marked those with an unclear diagnosis.")

# save df of boolean flags and grey/white matter tuples for each subject
csv_file = "data/subject_flags.csv"
print(f"Saving to CSV file {csv_file}")
flags = pd.DataFrame({"SUB_ID": unprocessed_data["SUB_ID"]})
flags["HAS_AXIAL"] = flags["SUB_ID"].isin(ax_ids["id"])
flags["HAS_SAGITTAL"] = flags["SUB_ID"].isin(sag_ids["id"])
flags["HAS_CORONAL"] = flags["SUB_ID"].isin(cor_ids["id"])
flags["AQ_OUTLIER"] = flags["SUB_ID"].isin(aq_outliers["SUB_ID"])
flags["UNCLEAR_DIAG"] = flags["SUB_ID"].isin(unclear_diag["SUB_ID"])
flags["AXIAL_GREY_MATTER"] = flags["SUB_ID"].map(dict(zip(ax_ids["id"], ax_ids["grey"])))
flags["AXIAL_WHITE_MATTER"] = flags["SUB_ID"].map(dict(zip(ax_ids["id"], ax_ids["white"])))
flags["SAGITTAL_GREY_MATTER"] = flags["SUB_ID"].map(dict(zip(sag_ids["id"], sag_ids["grey"])))
flags["SAGITTAL_WHITE_MATTER"] = flags["SUB_ID"].map(dict(zip(sag_ids["id"], sag_ids["white"])))
flags["CORONAL_GREY_MATTER"] = flags["SUB_ID"].map(dict(zip(cor_ids["id"], cor_ids["grey"])))
flags["CORONAL_WHITE_MATTER"] = flags["SUB_ID"].map(dict(zip(cor_ids["id"], cor_ids["white"])))
flags.to_csv(csv_file, index=False)
