'''
    Split the train/test/val populations

    Can select specific subsets of data to either withold or train on exclusively:
    - men
    - women
    - those with an unclear diagnosis
    - those with an AQ_TOTAL score which is an outlier
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MIN_STRATUM_SIZE = 6      # site/diagnosis groups smaller than this are merged when splitting


# --------------------------------------------------------------------------- #
# Function to Split Data
# --------------------------------------------------------------------------- #
def make_split(
        manifest: pd.DataFrame, 
        seed: int,
        tags: list[str] | None = None, 
        select_data: str | None = None,
        holdout: bool = False) -> pd.DataFrame:
    """
    Add a 'split' column. We split SUBJECTS (not scans), stratified on diagnosis and site so
    each set has a similar mix. If a site/diagnosis group is too small to stratify, we merge
    small groups, and if that still fails we stratify on diagnosis only. We also include ways
    to add other tags to the split, e.g. if we want to train on only men or only women.

    @param manifest:    a dataframe with columns "sub_id", "site", and "label"
    @param seed:
    @param tags:        list of what type of subjects we want to use to train the model
    @param select_data: if we want to select a specific subset of the data to train on.
                            - "men": only men
                            - "women": only women
                            - "unclear": only those with an unclear diagnosis
                            - "outlier": only those with an AQ_TOTAL score that is an outlier
    @param holdout:     if True, the select_data subset becomes an evaluation-only group instead:
                        those subjects get split "holdout" and everyone else is split into train/val/test.

    @return: a copy of the manifest with an added "split" column, used for train/val/test of model
    """

    if tags:
        tag_arr = ["label", "site", "age"] + tags
    else:
        tag_arr = ["label", "site", "age"]
    
    subj_all = manifest.drop_duplicates("sub_id").reset_index(drop=True)

    if select_data == "men":
        selected = subj_all["sex"] == 1
    elif select_data == "women":
        selected = subj_all["sex"] == 2
    elif select_data == "unclear":
        selected = subj_all["unclear_diag"] == True
    elif select_data == "outlier":
        selected = subj_all["aq_outlier"] == True
    elif select_data is None:
        selected = pd.Series(True, index=subj_all.index)
    else:
        raise ValueError(f"unknown select_data: {select_data!r}")

    if holdout:
        if select_data is None:
            raise ValueError("holdout=True needs a select_data subset to hold out")
        held_out = subj_all[selected][["sub_id"]].assign(split="holdout")
        subj = subj_all[~selected]
    else:
        held_out = subj_all.iloc[0:0][["sub_id"]].assign(split="holdout")
        subj = subj_all[selected]
    subj = subj.reset_index(drop=True)                       # idx below is positional

    strata = subj[tag_arr].astype(str).agg("_".join, axis=1)

    sizes = strata.map(strata.value_counts())
    strata = strata.where(sizes >= MIN_STRATUM_SIZE, subj["label"].astype(str) + "_rare")

    def stratified(idx, test_size):
        for col in (strata, subj["label"]):                 # try site+label, then label only
            try:
                return train_test_split(idx, test_size=test_size,
                                        stratify=col.iloc[idx], random_state=seed)
            except ValueError:
                print("  stratification failed; trying a coarser grouping")
        raise ValueError("could not stratify the split")

    idx = np.arange(len(subj))
    trainval_idx, test_idx = stratified(idx, TEST_FRAC)
    _, val_idx = stratified(trainval_idx, VAL_FRAC / (1 - TEST_FRAC))

    subj["split"] = "train"
    subj.loc[val_idx, "split"] = "val"
    subj.loc[test_idx, "split"] = "test"
    subj["split_seed"] = [seed] * len(subj["split"])

    # inner merge drops scans from subjects that are in neither the split nor the holdout group
    splits = pd.concat([subj[["sub_id", "split", "split_seed"]], held_out], ignore_index=True)
    out = manifest.drop(columns=["split", "split_seed"], errors="ignore").merge(splits, on="sub_id", how="inner")
    assert out.groupby("sub_id")["split"].nunique().max() == 1, "a subject ended up in two splits"
    return out