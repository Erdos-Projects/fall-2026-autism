"""
Tests for src/features/transformers.py, plus the orientation / shape bookkeeping
that ties transformers.py, preprocessing.py and train_baseline.py together.

No preprocessed .pt files are required: the tensor tests build small NIfTI volumes
in memory and push them through the real nii_to_tensor() and extract_zip() code
paths. The one test that does need real .pt files skips cleanly when they are absent.
"""

import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff

from features.preprocessing import get_scan_file
from features.transformers import (
    CLIP_PERCENTILE,
    FG_PERCENTILE,
    extract_zip,
    get_remove_set,
    nii_to_tensor,
)
from utils import parse_target_size, run_dir, save_pt_settings


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def nifti_bytes(array, affine=None):
    """Serialise a numpy volume as NIfTI bytes, as they appear inside the zip."""
    img = nib.Nifti1Image(
        np.asarray(array, dtype=np.float32),
        np.eye(4) if affine is None else np.asarray(affine, dtype=float),
    )
    return img.to_bytes()


def volume(shape=(20, 24, 28), seed=0):
    """A random positive volume, large enough to clear the foreground check.

    nii_to_tensor() rejects scans with fewer than 1000 foreground voxels, so
    fixtures must be substantially bigger than that.
    """
    rng = np.random.RandomState(seed)
    return rng.rand(*shape).astype(np.float32)


def reoriented(array, codes):
    """The same anatomy, stored with the given NIfTI axis codes.

    Built with nibabel's own orientation helpers, so the affine is guaranteed
    consistent with the reordered array. Hand-building this (a flipped array plus
    a negated affine) is easy to get subtly wrong.
    """
    ornt = axcodes2ornt(codes)
    stored = apply_orientation(array, ornt)
    return nifti_bytes(stored, inv_ornt_aff(ornt, array.shape))


def as_posix(path):
    """Normalise separators so the tests behave the same on Windows."""
    return str(path).replace("\\", "/")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def pt_files(repo_root):
    """Real preprocessed tensors, if they happen to be present locally."""
    scan_dir = repo_root / "data" / "scans" / "mri_sagittal" / "96_96_96"
    if not scan_dir.is_dir() or not any(scan_dir.glob("*.pt")):
        pytest.skip("no preprocessed .pt files; run run.sh -e to generate them")
    return sorted(scan_dir.glob("*.pt"))


# --------------------------------------------------------------------------- #
# Tensor shape and dtype
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target_shape", [
    (96, 96, 96),      # the project default
    (64, 64, 64),
    (32, 40, 48),      # non-cubic: catches a hardcoded 96
])
def test_nii_to_tensor_shape_and_dtype(target_shape):
    """Output must be (1, *TARGET_SHAPE) fp16 -- what Conv3d expects and what
    ScanDataset.__getitem__ unpacks."""
    tensor = nii_to_tensor(nifti_bytes(volume()), "synthetic", target_shape)

    assert tensor is not None, "valid volume was rejected by the foreground check"
    assert tuple(tensor.shape) == (1, *target_shape)
    assert tensor.dtype == torch.float16


def test_shape_matches_manifest_recorded_shape(manifest):
    """The 'img shape' column and the shape in pt_path must agree, otherwise the
    manifest describes tensors that do not exist on disk."""
    recorded = parse_target_size(manifest["img shape"].iloc[0])
    assert recorded == (96, 96, 96)

    path = as_posix(manifest["pt_path"].iloc[0])
    assert "/96_96_96/" in path, f"pt_path is not in the 96_96_96 folder: {path}"


def test_empty_foreground_returns_none():
    """All-zero volumes must be skipped rather than producing a garbage tensor."""
    assert nii_to_tensor(nifti_bytes(np.zeros((20, 24, 28))), "blank", (96, 96, 96)) is None


# --------------------------------------------------------------------------- #
# Intensity normalisation
# --------------------------------------------------------------------------- #
def test_background_is_zeroed_and_foreground_standardised():
    """nii_to_tensor z-scores the foreground and sets everything else to 0.

    Uses a target shape equal to the input shape so trilinear resampling is a
    no-op. Measuring the z-score on an upsampled volume does not work: going from
    20x24x28 to 96^3 averages neighbouring voxels together and blends the masked
    foreground into the zeroed background, which pushes the foreground std well
    below 1 regardless of whether the normalisation is correct.
    """
    tensor = nii_to_tensor(
        nifti_bytes(volume(shape=(24, 24, 24))), "synthetic", (24, 24, 24)
    )
    arr = tensor.float().numpy()

    foreground = arr[arr != 0]

    assert (arr == 0).any(), "no background voxels; the mask did nothing"
    assert foreground.size > 0
    assert abs(float(foreground.mean())) < 0.05, "foreground is not centred on 0"
    assert 0.85 < float(foreground.std()) < 1.15, "foreground is not unit-scaled"

def test_large_raw_intensities_do_not_overflow_fp16():
    """Normalisation must happen before the fp16 cast.

    If the cast ever moves ahead of the z-score, raw scanner intensities above
    65504 become inf. A volume scaled up by 1e5 reproduces that.
    """
    tensor = nii_to_tensor(nifti_bytes(volume() * 1e5), "bright", (48, 48, 48))

    assert tensor is not None
    assert torch.isfinite(tensor.float()).all(), "fp16 overflow produced inf/NaN"
    assert np.abs(tensor.float().numpy()).max() < 100, "values are not normalised"
# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #
def test_orientation_is_normalised_before_resizing():
    """The same anatomy stored in two different orientations must produce the
    same tensor.

    This is the guarantee as_closest_canonical() provides, and the reason scans
    from different sites are comparable at all. If that call is ever removed,
    site-to-site performance would silently degrade while every other test passed.
    """
    arr = volume()
    ras = nii_to_tensor(nifti_bytes(arr), "ras", (48, 48, 48))
    las = nii_to_tensor(reoriented(arr, ("L", "A", "S")), "las", (48, 48, 48))

    assert ras is not None and las is not None
    assert torch.allclose(ras.float(), las.float(), atol=0.2), (
        "stored orientation changed the tensor; as_closest_canonical may have been dropped"
    )


def test_manifest_orientation_matches_scan_folder(manifest):
    """Every pt_path in the sagittal manifest must point into the sagittal folder."""
    wrong = [
        p for p in manifest["pt_path"]
        if "/mri_sagittal/" not in as_posix(p)
    ]
    assert not wrong, f"{len(wrong)} paths point outside mri_sagittal/, e.g. {wrong[:3]}"


@pytest.mark.parametrize("orientation", ["axial", "sagittal", "coronal"])
def test_run_dir_encodes_orientation_shape_and_seed(orientation):
    """Results must land in results/{orientation}/{shape}/{seed}, so runs with
    different settings can never overwrite each other."""
    path = run_dir(orientation, (96, 96, 96), 42)

    parts = as_posix(path).split("/")
    assert orientation in parts
    assert "96_96_96" in parts
    assert "42" in parts


# --------------------------------------------------------------------------- #
# Settings round-trip
# --------------------------------------------------------------------------- #
def test_saved_settings_record_orientation_and_shape(tmp_path):
    """settings.json is the audit trail for how a set of .pt files was built."""
    import json

    target = (64, 64, 64)
    path = save_pt_settings(tmp_path, "coronal", target, FG_PERCENTILE, CLIP_PERCENTILE)
    settings = json.loads(path.read_text())

    assert settings["orientation"] == "coronal"
    assert settings["target_shape"] == list(target)
    assert settings["normalization"]["fg_percentile"] == list(FG_PERCENTILE)
    assert settings["normalization"]["clip_percentile"] == CLIP_PERCENTILE


@pytest.mark.parametrize("text,expected", [
    ("(96, 96, 96)", (96, 96, 96)),   # how manifest_sagittal.csv stores it
    ("96,96,96",     (96, 96, 96)),   # how run.sh passes it
    ("64 64 64",     (64, 64, 64)),
])
def test_parse_target_size_round_trips(text, expected):
    assert parse_target_size(text) == expected


# --------------------------------------------------------------------------- #
# Scan-file and subject matching
# --------------------------------------------------------------------------- #
def test_get_scan_file_matches_whole_tokens_only():
    """get_scan_file matches on '_' tokens. Subject 5 must not match a file
    belonging to subject 55, or labels get attached to the wrong scan."""
    names = [
        "ABIDE_55_MRI_x_I900001.pt",
        "ABIDE_51456_MRI_x_I329274.pt",
    ]

    assert get_scan_file(51456, names) == "ABIDE_51456_MRI_x_I329274.pt"
    assert get_scan_file(5, names) is None
    assert get_scan_file(99999, names) is None


def test_get_remove_set_selects_stimulant_subjects():
    """transformers.py skips subjects flagged as on stimulants at scan time."""
    data = pd.DataFrame({
        "SUB_ID": [1, 2, 3, 4],
        "OFF_STIMULANTS_AT_SCAN": [0, 1, 0, 1],
    })

    assert get_remove_set(data) == {"1", "3"}


# --------------------------------------------------------------------------- #
# End-to-end extraction
# --------------------------------------------------------------------------- #
def test_extract_zip_end_to_end(tmp_path):
    """Full path: zip member -> filename parse -> inventory lookup -> .pt file.

    This is the one test that covers the subject-id/label linkage, which is where
    a silently wrong label would come from -- and a wrong label would poison every
    downstream result without any other test noticing.
    """
    zip_path = tmp_path / "mri_sagittal.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for sub in (101, 102):
            zf.writestr(
                f"ABIDE_{sub}_MRI_MP-RAGE_x_I9{sub}.nii",
                nifti_bytes(volume()),
            )

    inventory = pd.DataFrame({"SUB_ID": [101, 102], "DX_GROUP": [1, 2]})
    extract_zip(
        zip_path, inventory, remove_set=set(),
        orientation="sagittal", target_shape=(48, 48, 48),
    )

    save_dir = tmp_path / "mri_sagittal" / "48_48_48"
    saved = sorted(save_dir.glob("*.pt"))
    assert len(saved) == 2, f"expected 2 tensors, found {len(saved)} in {save_dir}"

    # DX_GROUP 1 (autism) -> label 1; DX_GROUP 2 (control) -> label 0
    first = torch.load(saved[0], map_location="cpu")
    second = torch.load(saved[1], map_location="cpu")

    assert first["id"] == 101 and first["label"] == 1
    assert second["id"] == 102 and second["label"] == 0
    for payload in (first, second):
        assert tuple(payload["vol"].shape) == (1, 48, 48, 48)


def test_extract_zip_skips_removed_subjects(tmp_path):
    """Subjects in remove_set must not produce tensors at all."""
    zip_path = tmp_path / "mri_sagittal.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for sub in (101, 102):
            zf.writestr(
                f"ABIDE_{sub}_MRI_MP-RAGE_x_I9{sub}.nii",
                nifti_bytes(volume()),
            )

    inventory = pd.DataFrame({"SUB_ID": [101, 102], "DX_GROUP": [1, 2]})
    extract_zip(
        zip_path, inventory, remove_set={"101"},
        orientation="sagittal", target_shape=(48, 48, 48),
    )

    saved = sorted((tmp_path / "mri_sagittal" / "48_48_48").glob("*.pt"))
    assert len(saved) == 1
    assert torch.load(saved[0], map_location="cpu")["id"] == 102


# --------------------------------------------------------------------------- #
# Real .pt files, when present locally
# --------------------------------------------------------------------------- #
def test_real_tensors_have_expected_shape(pt_files):
    """Spot-check actual preprocessed tensors.

    Skips on a clean clone; runs on a machine that has run run.sh -e.
    """
    for path in pt_files[:20]:
        payload = torch.load(path, map_location="cpu")
        assert tuple(payload["vol"].shape) == (1, 96, 96, 96), path.name
        assert payload["vol"].dtype == torch.float16, path.name
        assert payload["label"] in (0, 1), path.name
        assert isinstance(payload["id"], int), path.name