'''
    Main preprocessing pipeline that drops unused features, integrates engineered features, and can be reused in modeling.
'''

import csv
import pandas as pd
from pathlib import Path


unprocessed_data = pd.read_csv("data/data_inventory.csv")
control = unprocessed_data[unprocessed_data["DX_GROUP"] == 2]
autism = unprocessed_data[unprocessed_data["DX_GROUP"] == 1]


def checkScanType(dir: str) -> list:
    all_ids = unprocessed_data["SUB_ID"].unique()
    subj = []
    names = [p.name for p in Path(dir).iterdir() if p.is_file() and p.suffix == ".pt"]
    for id in all_ids:
        for name in names:
            if str(id) in name.split("_"):
                subj.append(id)
                break
    return subj


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
print(f"replaced {n_replaced} {bad_value} values with empty values in {inventory_path}")

# now to save ids we want to flag for specific circumstances
# scan types for each subject
ax_ids = checkScanType("data/scans/mri_axial")
sag_ids = checkScanType("data/scans/mri_sagittal")
cor_ids = checkScanType("data/scans/mri_coronal")

# get aq outliers
aq_border = 19
aq_outliers = pd.concat([
    control[control["AQ_TOTAL"] >= aq_border],
    autism[autism["AQ_TOTAL"] <= aq_border],
])

# get subjects with unclear diagnoses
unclear_diag = unprocessed_data[(unprocessed_data["DSM_IV_TR"] >= 3)]

# save df of boolean flags for each subject
flags = pd.DataFrame({"SUB_ID": unprocessed_data["SUB_ID"]})
flags["HAS_AXIAL"] = flags["SUB_ID"].isin(ax_ids)
flags["HAS_SAGITTAL"] = flags["SUB_ID"].isin(sag_ids)
flags["HAS_CORONAL"] = flags["SUB_ID"].isin(cor_ids)
flags["AQ_OUTLIER"] = flags["SUB_ID"].isin(aq_outliers["SUB_ID"])
flags["UNCLEAR_DIAG"] = flags["SUB_ID"].isin(unclear_diag["SUB_ID"])
flags.to_csv("data/subject_flags.csv", index=False)

