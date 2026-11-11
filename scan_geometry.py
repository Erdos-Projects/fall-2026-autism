"""
Reports the native geometry of every scan in data/MRI, to answer: which scans are real 3D
volumes (any plane can be reformatted) and which are thick-slice acquisitions that are only
trustworthy in one plane?

Reads only the NIfTI headers (no image data), so it is fast. For each file it records:
  - array shape and voxel size in mm, in anatomical order (x = left-right, y = front-back,
    z = up-down), worked out from the affine so it does not matter how the file is stored
  - the storage orientation codes, e.g. RAS or LPS
  - ratio = largest voxel side / smallest voxel side
  - native_plane: "3D" if ratio < ISO_RATIO (roughly isotropic voxels), otherwise the plane
    perpendicular to the thickest voxel side: axial (thick along z), coronal (thick along y),
    or sagittal (thick along x). That is the plane the slices were acquired in.

Writes data/scan_geometry.csv and prints a summary, including a site x native_plane table if
data/data_inventory.csv is available (acquisition type is usually decided per site).

Run from the project folder:   python scan_geometry.py
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

MRI_DIR = Path("data/MRI")
INVENTORY_PATH = Path("data/data_inventory.csv")
OUT_PATH = Path("data/scan_geometry.csv")

ISO_RATIO = 1.5          # largest/smallest voxel side below this counts as "3D" (isotropic-ish)
THICK_AXIS_TO_PLANE = {"x": "sagittal", "y": "coronal", "z": "axial"}


def anatomical_axis(code):
    """Axis code letter -> anatomical axis: L/R -> x, A/P -> y, I/S -> z."""
    return "x" if code in ("L", "R") else "y" if code in ("A", "P") else "z"


def describe(path):
    img = nib.load(path)                                    # header only; data is not read
    codes = nib.aff2axcodes(img.affine)                     # direction of each array axis
    voxel = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))  # voxel size along each array axis (mm)
    shape = img.shape[:3]

    by_axis = {anatomical_axis(c): (float(v), int(s)) for c, v, s in zip(codes, voxel, shape)}
    vox = {a: by_axis[a][0] for a in "xyz"}
    n = {a: by_axis[a][1] for a in "xyz"}

    ratio = max(vox.values()) / max(min(vox.values()), 1e-6)
    thick = max(vox, key=vox.get)
    plane = "3D" if ratio < ISO_RATIO else THICK_AXIS_TO_PLANE[thick]

    return dict(ndim=len(img.shape), axcodes="".join(codes),
                n_x=n["x"], n_y=n["y"], n_z=n["z"],
                vox_x=round(vox["x"], 3), vox_y=round(vox["y"], 3), vox_z=round(vox["z"], 3),
                voxel_mm=f"{vox['x']:.1f} x {vox['y']:.1f} x {vox['z']:.1f}",
                ratio=round(ratio, 2), native_plane=plane)


def main():
    files = sorted(p for p in MRI_DIR.rglob("*") if p.name.endswith((".nii", ".nii.gz")))
    print(f"found {len(files)} files under {MRI_DIR}")

    rows = []
    for p in files:
        parts = p.name.split("_")                           # I329309_50357_MP-RAGE.nii
        row = {"file": str(p), "scan_id": parts[0],
               "sub_id": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None}
        try:
            row.update(describe(p))
        except Exception as e:                              # a bad file should not stop the survey
            print(f"FAILED {p}: {e}")
            continue
        rows.append(row)
    df = pd.DataFrame(rows)

    if INVENTORY_PATH.exists():
        inv = pd.read_csv(INVENTORY_PATH, na_values=["-9999"])[["SUB_ID", "SITE_ID", "DX_GROUP"]]
        inv = inv.rename(columns={"SUB_ID": "sub_id", "SITE_ID": "site", "DX_GROUP": "dx_group"})
        df = df.merge(inv, on="sub_id", how="left")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n{len(df)} scans, {df['sub_id'].nunique()} subjects. Wrote {OUT_PATH}")
    if (df["ndim"] > 3).any():
        print(f"WARNING: {(df['ndim'] > 3).sum()} files have more than 3 dimensions (4D); "
              "check these, they may not be plain T1 volumes")

    print("\nNative plane (3D = roughly isotropic voxels, so any plane can be reformatted):")
    print(df["native_plane"].value_counts().to_string())

    print("\nStorage orientation codes (how the array axes are stored):")
    print(df["axcodes"].value_counts().to_string())

    print("\nMost common voxel sizes in mm (x, y, z):")
    print(df["voxel_mm"].value_counts().head(12).to_string())

    if "site" in df.columns:
        print("\nSubjects per site and native plane:")
        sub = df.drop_duplicates(["sub_id", "native_plane"])
        print(pd.crosstab(sub["site"].fillna("unknown"), sub["native_plane"]).to_string())
        print("\nAutism fraction by native plane (DX_GROUP == 1), per scan:")
        print(df.assign(autism=(df["dx_group"] == 1)).groupby("native_plane")["autism"]
                .agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
