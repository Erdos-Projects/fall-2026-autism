"""
Tests that the manifest agrees with the data inventory.

The manifest is the contract between features/preprocessing.py and train_baseline.py:
if a label, a subject id, or a scan id is wrong here, every downstream result is wrong
and nothing else will catch it.
"""

from pathlib import Path

import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Manifest vs. inventory
# --------------------------------------------------------------------------- #
def test_manifest_labels_match_inventory(manifest, inventory):
    """Every label in the manifest must match DX_GROUP in the inventory.

    ABIDE codes DX_GROUP as 1 = autism, 2 = control; the manifest recodes to
    1 = autism, 0 = control.
    """
    merged = manifest[["sub_id", "label"]].merge(
        inventory[["SUB_ID", "DX_GROUP"]],
        left_on="sub_id", right_on="SUB_ID", how="left",
    )

    assert merged["DX_GROUP"].notna().all(), (
        "manifest contains subjects that are not in the inventory: "
        f"{merged.loc[merged['DX_GROUP'].isna(), 'sub_id'].tolist()[:10]}"
    )

    expected = merged["DX_GROUP"].map({1: 1, 2: 0})
    mismatches = merged[expected != merged["label"]]
    assert mismatches.empty, (
        f"{len(mismatches)} manifest rows disagree with DX_GROUP:\n"
        f"{mismatches.head().to_string()}"
    )


def test_manifest_labels_are_binary(manifest):
    """The label column feeds BCEWithLogitsLoss, so it must be 0/1 only."""
    assert set(manifest["label"].unique()) <= {0, 1}


def test_manifest_has_one_row_per_subject(manifest):
    """One scan per participant; the per-subject averaging in train_baseline.py
    assumes this."""
    duplicated = manifest.loc[manifest["sub_id"].duplicated(), "sub_id"].tolist()
    assert not duplicated, f"subjects with more than one row: {duplicated[:10]}"


def test_every_manifest_subject_is_in_inventory(manifest, inventory):
    unknown = set(manifest["sub_id"]) - set(inventory["SUB_ID"])
    assert not unknown, f"manifest subjects missing from inventory: {sorted(unknown)[:10]}"


# --------------------------------------------------------------------------- #
# Scan ids vs. participant ids
# --------------------------------------------------------------------------- #
def test_scan_ids_match_participant_ids(manifest):
    """The subject id has to appear in the scan filename, and scan_id must be the
    filename's last token.

    This is what features/preprocessing.py:get_scan_file relies on to decide which
    .pt file belongs to which participant, and what transformers.py writes into it.
    """
    problems = []
    for row in manifest.itertuples():
        stem_parts = Path(row.pt_path).stem.split("_")
        if str(row.scan_id) != stem_parts[-1]:
            problems.append(f"{row.sub_id}: scan_id={row.scan_id} != {stem_parts[-1]}")
        elif str(row.sub_id) not in stem_parts:
            problems.append(f"{row.sub_id}: subject id not in {Path(row.pt_path).name}")

    assert not problems, f"{len(problems)} scan id problems:\n" + "\n".join(problems[:10])


def test_pt_paths_are_relative(manifest):
    """Manifest paths must be repo-relative so the pipeline runs on any machine
    (run.sh sets PROJECT_DIR, and preprocessing.py joins paths against it)."""
    absolute = manifest.loc[
        manifest["pt_path"].str.startswith("/"), "pt_path"
    ].tolist()
    assert not absolute, f"{len(absolute)} absolute paths in manifest, e.g. {absolute[:3]}"


# --------------------------------------------------------------------------- #
# build_manifest() behaviour, on a small synthetic inventory
# --------------------------------------------------------------------------- #
def test_build_manifest_maps_labels_and_flags(monkeypatch):
    """Check the DX mapping and the AQ / unclear-diagnosis flags directly."""
    import features.preprocessing as pp

    inventory = pd.DataFrame({
        "SUB_ID": [1, 2, 3, 4],
        "SITE_ID": ["NYU"] * 4,
        "DX_GROUP": [1, 2, 1, 2],     # autism, control, autism, control
        "AGE_AT_SCAN": [10.0, 11.0, 12.0, 13.0],
        "SEX": [1, 2, 1, 2],
        "AQ_TOTAL": [25, 10, 10, 25],
        "DSM_IV_TR": [1, 1, 3, 2],
    })

    # build_manifest() looks for .pt files on disk; stub that out so the test
    # exercises only the label/flag logic.
    monkeypatch.setattr(pp, "get_pt_info", lambda sub_id, orientation, shape: {
        "path": f"data/scans/mri_sagittal/96_96_96/ABIDE_{sub_id}_x_I{sub_id}.pt",
        "scan_id": f"I{sub_id}",
    })

    out = pp.build_manifest(inventory, "sagittal", (96, 96, 96))

    assert out["label"].tolist() == [1, 0, 1, 0]

    # AQ outlier = control with AQ_TOTAL >= 19, or autism with AQ_TOTAL <= 19
    assert out["aq_outlier"].tolist() == [False, False, True, True]

    # unclear diagnosis = DSM_IV_TR >= 3
    assert out["unclear_diag"].tolist() == [False, False, True, False]


def test_clean_inventory_replaces_bad_values(tmp_path):
    """-9999 means 'missing' in ABIDE and must become empty, leaving the original
    file untouched."""
    import features.preprocessing as pp

    src = tmp_path / "raw.csv"
    dst = tmp_path / "clean.csv"
    src.write_text("SUB_ID,AQ_TOTAL\n1,-9999\n2,25\n", encoding="utf-8")

    pp.clean_inventory(src, dst, bad_value="-9999")

    out = pd.read_csv(dst)
    assert pd.isna(out.loc[0, "AQ_TOTAL"])
    assert out.loc[1, "AQ_TOTAL"] == 25
    assert "-9999" in src.read_text(encoding="utf-8"), "raw inventory was modified"