"""
Tests for src/splits/make_split.py.

The split is the single most important correctness property in this project: every
reported AUC and accuracy is only meaningful if no participant's scan appears in both
the training set and an evaluation set. These tests are mostly about that invariant,
plus the holdout / subset selection modes that make_split supports.
"""

import pandas as pd
import pytest

from splits.make_split import make_split

SPLIT_COLUMNS = {"split", "split_seed"}
ALL_SPLITS = {"train", "val", "test"}


# --------------------------------------------------------------------------- #
# No leakage between splits
# --------------------------------------------------------------------------- #
def test_no_subject_appears_in_more_than_one_split(manifest):
    """The headline invariant: each participant is in exactly one split."""
    split = make_split(manifest, seed=42)

    per_subject = split.groupby("sub_id")["split"].nunique()
    offenders = per_subject[per_subject > 1]
    assert offenders.empty, (
        f"{len(offenders)} subjects appear in multiple splits: {offenders.index[:10].tolist()}"
    )


def test_every_subject_gets_exactly_one_split(manifest):
    """No participant should be silently dropped or left unassigned."""
    split = make_split(manifest, seed=42)

    assert split["split"].notna().all()
    assert set(split["split"].unique()) <= ALL_SPLITS

    expected = set(manifest["sub_id"])
    assert set(split["sub_id"]) == expected, (
        f"subjects lost from split: {sorted(expected - set(split['sub_id']))[:10]}"
    )


def test_splits_are_disjoint_and_complete(manifest):
    """The three sets must partition the participants."""
    split = make_split(manifest, seed=42)

    by_split = {
        name: set(group["sub_id"])
        for name, group in split.groupby("split")
    }

    assert by_split["train"] & by_split["val"] == set()
    assert by_split["train"] & by_split["test"] == set()
    assert by_split["val"] & by_split["test"] == set()

    union = by_split["train"] | by_split["val"] | by_split["test"]
    assert union == set(manifest["sub_id"])


def test_no_subject_in_two_splits_synthetic(synthetic_manifest):
    """Same invariant, on a controlled cohort where every subject is reachable."""
    split = make_split(synthetic_manifest(), seed=7)

    per_subject = split.groupby("sub_id")["split"].nunique()
    assert per_subject.max() == 1


# --------------------------------------------------------------------------- #
# Proportions and stratification
# --------------------------------------------------------------------------- #
def test_split_proportions_are_approximately_15_15_70(manifest):
    """VAL_FRAC and TEST_FRAC are both 0.15, so train should be about 70%.

    Generous tolerance: stratified splitting plus the rare-group merging in
    make_split() means exact proportions are not guaranteed.
    """
    split = make_split(manifest, seed=42)
    subject_level = split.drop_duplicates("sub_id")

    fracs = subject_level["split"].value_counts(normalize=True)

    assert fracs.get("test", 0) == pytest.approx(0.15, abs=0.05)
    assert fracs.get("val", 0) == pytest.approx(0.15, abs=0.05)
    assert fracs.get("train", 0) == pytest.approx(0.70, abs=0.05)


def test_class_balance_is_preserved_across_splits(manifest):
    """Stratification should keep the autism fraction roughly equal in each split."""
    split = make_split(manifest, seed=42)
    subject_level = split.drop_duplicates("sub_id")

    overall = subject_level["label"].mean()
    per_split = subject_level.groupby("split")["label"].mean()

    for name, frac in per_split.items():
        assert frac == pytest.approx(overall, abs=0.10), (
            f"{name} autism fraction {frac:.2f} far from overall {overall:.2f}"
        )


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_gives_same_split(synthetic_manifest):
    """Re-running with the same seed must reproduce the split exactly, so a
    reported result can be regenerated."""
    cohort = synthetic_manifest()

    first = make_split(cohort, seed=42)[["sub_id", "split"]].sort_values("sub_id")
    second = make_split(cohort, seed=42)[["sub_id", "split"]].sort_values("sub_id")

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_different_seed_gives_different_split(synthetic_manifest):
    """A different seed should actually reshuffle, otherwise the seed is being
    ignored and reported variance across seeds would be meaningless."""
    cohort = synthetic_manifest()

    first = make_split(cohort, seed=42).set_index("sub_id")["split"]
    second = make_split(cohort, seed=43).set_index("sub_id")["split"]

    assert not first.equals(second)


def test_split_seed_is_recorded(synthetic_manifest):
    """The saved split records which seed produced it, for the audit trail."""
    split = make_split(synthetic_manifest(), seed=123)
    assert (split["split_seed"] == 123).all()


# --------------------------------------------------------------------------- #
# Subset selection and holdout mode
# --------------------------------------------------------------------------- #
def test_holdout_group_is_isolated(manifest):
    """holdout=True should set the selected subjects aside and never let them
    reach train/val/test."""
    split = make_split(manifest, seed=42, select_data="unclear", holdout=True)

    holdout_ids = set(split.loc[split["split"] == "holdout", "sub_id"])
    assert holdout_ids, "holdout=True produced no holdout subjects"

    trained_on = set(split.loc[split["split"].isin(ALL_SPLITS), "sub_id"])
    assert holdout_ids & trained_on == set(), (
        f"{len(holdout_ids & trained_on)} holdout subjects leaked into train/val/test"
    )

    # Everyone must still be accounted for
    assert holdout_ids | trained_on == set(split["sub_id"])


def test_holdout_requires_select_data(manifest):
    """holdout=True without a subset is meaningless; make_split should refuse."""
    with pytest.raises(ValueError, match="holdout=True needs a select_data subset"):
        make_split(manifest, seed=42, holdout=True)


@pytest.mark.parametrize("select_data,column,value", [
    ("men", "sex", 1),
    ("women", "sex", 2),
    ("unclear", "unclear_diag", True),
    ("outlier", "aq_outlier", True),
])
def test_select_data_keeps_only_the_requested_subset(synthetic_manifest, select_data, column, value):
    """Without holdout, select_data should restrict training to just that subset.

    Uses the synthetic cohort so every subset is guaranteed non-empty; the real
    manifest has no aq_outlier subjects at all (see test below).
    """
    cohort = synthetic_manifest()

    n_selected = int((cohort[column] == value).sum())
    assert n_selected >= 40, (
        f"fixture only selects {n_selected} subjects for {select_data!r}; "
        "make_split needs ~40+ to stratify"
    )

    split = make_split(cohort, seed=42, select_data=select_data)

    assert set(split["split"].unique()) <= ALL_SPLITS
    assert (split[column] == value).all(), (
        f"select_data={select_data!r} let through rows where {column} != {value}"
    )

def test_no_holdout_rows_by_default(synthetic_manifest):
    """The default call must not create a holdout group."""
    split = make_split(synthetic_manifest(), seed=42)
    assert "holdout" not in set(split["split"].unique())


# --------------------------------------------------------------------------- #
# Re-splitting an already-split manifest
# --------------------------------------------------------------------------- #
def test_resplit_replaces_old_split_column(synthetic_manifest):
    """make_split drops any pre-existing 'split' column, so calling it twice
    yields one column of fresh assignments rather than duplicates."""
    once = make_split(synthetic_manifest(), seed=42)
    twice = make_split(once, seed=99)

    assert [c for c in twice.columns if c == "split"].__len__() == 1
    assert (twice["split_seed"] == 99).all()
    assert twice.groupby("sub_id")["split"].nunique().max() == 1

@pytest.mark.xfail(
    reason="aq_outlier is all False in the committed manifest: AQ_TOTAL is almost "
           "entirely missing in ABIDE I, so the >=19 / <=19 comparison never fires",
    strict=False,
)
def test_aq_outlier_flag_has_subjects(manifest):
    """If this ever passes, the AQ flag became usable and select_data='outlier'
    is a real experimental option."""
    n = int(manifest["aq_outlier"].sum())
    assert n > 0, (
        f"aq_outlier is all False across {len(manifest)} subjects; "
        "select_data='outlier' selects nothing and crashes inside make_split()."
    )


@pytest.mark.xfail(
    reason="make_split raises an opaque pandas error instead of a clear message "
           "when the selected subset is empty",
    strict=False,
)
def test_empty_selection_gives_clear_error(synthetic_manifest):
    """An empty selection is a user error that deserves a readable ValueError."""
    cohort = synthetic_manifest()
    cohort["aq_outlier"] = False

    with pytest.raises(ValueError, match="subject"):
        make_split(cohort, seed=42, select_data="outlier")