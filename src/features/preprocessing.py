'''
    Cleans data/data_inventory.csv and writes data/manifest_{ORIENTATION}.csv for train_baseline.py
    Author: Jess Miears


'''

import csv
import pandas as pd
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so src/utils.py can be imported
from utils import check_project_dir, parse_target_size, PROJECT_DIR

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
INVENTORY_PATH = Path(f"{PROJECT_DIR}/data/data_inventory.csv")
CLEAN_INVENTORY_PATH = Path(f"{PROJECT_DIR}/data/data_inventory_cleaned.csv")

BAD_VALUE = "-9999"
AQ_BORDER = 19


# --------------------------------------------------------------------------- #
# Clean Demographic Data
# --------------------------------------------------------------------------- #
def clean_inventory(inventory_path: Path, clean_path: Path, bad_value: str = BAD_VALUE) -> None:
    '''
        write a copy of the inventory with the bad values replaced by empty values,
        leaving the original file unchanged

        @param inventory_path : path to the raw data inventory csv
        @param clean_path : path to write the cleaned csv to
        @param bad_value : the placeholder value to remove
    '''
    with open(inventory_path, newline="") as f:
        rows = list(csv.reader(f))
    n_replaced = 0
    for row in rows:
        for j, value in enumerate(row):
            if value.strip() == bad_value:
                row[j] = ""
                n_replaced += 1
    with open(clean_path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    print(f"Replaced {n_replaced} {bad_value} values with empty values; saved to {clean_path}.")


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
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


def get_pt_info(subj_id: int, scan_type: str, target_shape: tuple[int]) -> dict:
    '''
        get the path to the .pt file for a given subject id and scan type

        @param subj_id : subject id
        @param scan_type : the orientation of the scan (either 'axial', 'sagittal', or 'coronal')

        @return path : path to the .pt file if it exists, else None
    '''
    check_project_dir(PROJECT_DIR)
    zip_path = Path(f"{PROJECT_DIR}/data/scans/mri_{scan_type}.zip")
    scan_dir = Path(f"{zip_path.with_suffix('')}/{'_'.join(map(str, target_shape))}")
    if not scan_dir.is_dir():
        raise FileNotFoundError(f"{scan_dir} does not exist! Run run.sh with flag -e to extract the scans.")
    names = [p.name for p in scan_dir.iterdir() if p.is_file() and p.suffix == ".pt"]
    file = get_scan_file(subj_id, names)
    if file:
        return {"path": str(scan_dir.relative_to(PROJECT_DIR) / file), "scan_id": Path(file).stem.split("_")[-1]}
    return {"path": None, "scan_id": None}


# --------------------------------------------------------------------------- #
# Make Manifest CSV
# --------------------------------------------------------------------------- #
def build_manifest(data: pd.DataFrame, orientation: str, target_shape: tuple[int]) -> pd.DataFrame:
    '''
        build the manifest of all important information for each subject that has a scan

        @param data : the cleaned data inventory
        @param orientation : the orientation of the scan (either 'axial', 'sagittal', or 'coronal')

        @return manifest : one row per subject with a scan in the given orientation
    '''
    # split into control and autism groups
    control = data[data["DX_GROUP"] == 2]
    autism = data[data["DX_GROUP"] == 1]

    # get file paths
    scan_info = [get_pt_info(subj_id, orientation, target_shape) for subj_id in data["SUB_ID"]]

    # get aq outliers
    aq_outliers = pd.concat([
        control[control["AQ_TOTAL"] >= AQ_BORDER],
        autism[autism["AQ_TOTAL"] <= AQ_BORDER],
    ])

    # get subjects with unclear diagnoses
    unclear_diag = data[(data["DSM_IV_TR"] >= 3)]

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
            "pt_path": [p["path"] for p in scan_info],
            "img shape": [target_shape] * len(data["SUB_ID"])
        }
    )

    # remove subjects with no scans
    return manifest.dropna(subset=["scan_id"], how="all")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--o", "-orientation", default="sagittal",
                    help="set scan orientation")
    ap.add_argument("--shape", default="(96,96,96)",
                    help="set image shape (format is (int, int, int))")
    args = ap.parse_args()

    target_shape = parse_target_size(args.shape)
    orientation = args.o
    
    check_project_dir(PROJECT_DIR)
    csv_file = f"{PROJECT_DIR}/data/manifest_{orientation}.csv"

    clean_inventory(INVENTORY_PATH, CLEAN_INVENTORY_PATH)
    data = pd.read_csv(CLEAN_INVENTORY_PATH)
    manifest = build_manifest(data, orientation, target_shape)

    print(f"Saving built manifest to CSV file {csv_file}")
    manifest.to_csv(csv_file, index=False)


if __name__ == "__main__":
    main()
