'''
    Small helpers shared by the pipeline scripts.

    Author: Jess Miears
'''

from datetime import datetime
from pathlib import Path
import json
import os
import re


# --------------------------------------------------------------------------- #
# Project Settings - set in run.sh
# --------------------------------------------------------------------------- #
PROJECT_DIR = os.environ.get("PROJECT_DIR") 

# --------------------------------------------------------------------------- #
# Helpful Functions
# --------------------------------------------------------------------------- #
def check_project_dir(project_dir: str | None) -> None:
    '''
        make sure PROJECT_DIR is set and points at the repo, so paths built from it are valid

        @param project_dir : the value of the PROJECT_DIR environment variable
    '''
    if not project_dir:
        raise ValueError("PROJECT_DIR is not set! Use run.sh, or export PROJECT_DIR=<path to the fall-2026-autism repo>.")
    if not (Path(project_dir) / "src").is_dir() or not (Path(project_dir) / "data").is_dir():
        raise ValueError(f"PROJECT_DIR={project_dir} is not the fall-2026-autism repo (no src/ or data/ folder).")


def parse_target_size(target_size: str | None) -> tuple[int, int, int]:
    '''
        turn the TARGET_SHAPE environment variable into a tuple of 3 integers
        (accepts e.g. "96,96,96", "96 96 96" or "(96, 96, 96)"; defaults to (96, 96, 96) when unset)

        @param target_size : the value of the TARGET_SHAPE environment variable

        @return target_shape : tuple of 3 positive integers
    '''
    if not target_size:
        return (96, 96, 96)
    parts = [p for p in re.split(r"[\s,]+", target_size.strip("()[] ")) if p]
    if len(parts) != 3 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"TARGET_SHAPE={target_size} is not 3 positive integers (e.g. '96,96,96').")
    return tuple(int(p) for p in parts)


def save_pt_settings(out_dir: str | Path, orientation: str, target_shape: tuple[int, int, int],
                                fg_percentile: tuple[float, float], clip_percentile: float) -> Path:
    '''
        write the settings used to make the .pt files to out_dir/settings.json

        @param out_dir : folder the .pt files are saved in
        @param orientation : the orientation of the scan (either 'axial', 'sagittal', or 'coronal')
        @param target_shape : tuple of 3 integers the scans were resized to
        @param fg_percentile : foreground threshold used for normalization, as (fraction, percentile)
        @param clip_percentile : percentile of foreground intensities the scans were clipped at

        @return settings_path : path of the written JSON file
    '''
    settings = {
        "orientation": orientation,
        "target_shape": list(target_shape),
        "normalization": {
            "fg_percentile": list(fg_percentile),
            "clip_percentile": clip_percentile,
        },
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    settings_path = Path(out_dir) / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=4))
    return settings_path


def save_run_settings(out_dir: str | Path, pt_dir: str | Path, seed: int) -> Path:
    '''
        write the settings used for a training run to out_dir/settings.json: the preprocessing
        settings saved next to the .pt files, plus the seed and the date of the run

        @param out_dir : results folder of the run
        @param pt_dir : folder of the .pt files the run trains on
        @param seed : random seed of the run

        @return settings_path : path of the written JSON file
    '''
    pt_settings_path = PROJECT_DIR / Path(pt_dir) / "settings.json"
    if not pt_settings_path.exists():
        raise ValueError(f"{pt_settings_path} does not exist! Run run.sh again with flag -e to remake the .pt files.")
    settings = json.loads(pt_settings_path.read_text())
    settings["preprocessing_date"] = settings.pop("date")
    settings["seed"] = seed
    settings["date"] = datetime.now().isoformat(timespec="seconds")
    settings_path = PROJECT_DIR / Path(out_dir) / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=4))
    return settings_path


def run_dir(orientation: str, target_shape: tuple[int, int, int], seed: int,
            smoke: bool = False, project_dir: str | Path | None = PROJECT_DIR) -> Path:
    '''
        build the results folder of a training run, where best_model.pt, manifest_split.csv,
        training_log.csv and settings.json are saved

        @param orientation : the orientation of the scan (either 'axial', 'sagittal', or 'coronal')
        @param target_shape : tuple of 3 integers the scans were resized to
        @param seed : random seed of the run
        @param smoke : True for a smoke test run (saved in data/smoke instead)
        @param project_dir : path to the repo; defaults to the PROJECT_DIR environment variable

        @return run_dir : path of the results folder
    '''
    check_project_dir(project_dir)
    if smoke:
        return Path(project_dir) / "data" / "smoke"
    return Path(project_dir) / "results" / orientation / "_".join(map(str, target_shape)) / str(seed)
