'''
    Small helpers shared by the pipeline scripts.

    Author: Jess Miears
'''

from pathlib import Path
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