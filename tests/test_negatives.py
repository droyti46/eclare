"""
Eclare — Evolutionary Contrastive Learning for Alignment of Regulatory Elements.

Evaluation utilities for cross-species regulatory element alignment.

Copyright (c) 2026 Eclare contributors.
"""

import pandas as pd
import pytest

from eclare.data.negatives import build_pospos_dataset


def test_excludes_all_positive_partners_and_moves_mouse_context():
    positives = pd.DataFrame({
        "human_id": ["H1", "H1", "H2"],
        "mouse_id": ["M1", "M2", "M3"],
        "tissue": ["liver"] * 3,
        "human_sequence": ["AAA", "AAA", "CCC"],
        "mouse_sequence": ["AAA", "CCC", "GGG"],
        "mouse_start": [10, 20, 30],
        "mouse_zscore_DNase": [1.0, 2.0, 3.0],
    })
    original = positives.copy(deep=True)
    result = build_pospos_dataset(positives, seed=42, progress=False)
    negative = result.query("label == 0")
    h1 = negative.query("human_id == 'H1'")
    assert len(result) == 6
    assert len(negative) == 3
    assert h1.mouse_id.tolist() == ["M3", "M3"]
    assert h1.mouse_sequence.tolist() == ["GGG", "GGG"]
    assert h1.mouse_start.tolist() == [30, 30]
    assert h1.mouse_zscore_DNase.tolist() == [3.0, 3.0]
    assert h1.human_sequence.tolist() == ["AAA", "AAA"]
    assert set(negative.negative_type) == {"pospos_shuffled_K1"}
    assert set(result.query("label == 1").negative_type) == {"positive"}
    pd.testing.assert_frame_equal(positives, original)
    pd.testing.assert_frame_equal(
        result, build_pospos_dataset(positives, seed=42, progress=False),
    )


def test_mouse_pool_and_features_are_tissue_specific():
    positives = pd.DataFrame({
        "human_id": ["H1", "H2", "H3", "H4"],
        "mouse_id": ["M1", "M2", "M3", "M4"],
        "tissue": ["liver", "liver", "brain", "brain"],
        "mouse_sequence": ["AA", "CC", "GG", "TT"],
    })
    result = build_pospos_dataset(positives, progress=False)
    negative = result.query("label == 0").set_index("human_id")
    assert negative.loc[["H1", "H2", "H3", "H4"], "mouse_id"].tolist() == [
        "M2", "M1", "M4", "M3",
    ]
    assert negative.loc[["H1", "H2", "H3", "H4"], "mouse_sequence"].tolist() == [
        "CC", "AA", "TT", "GG",
    ]


def test_empty_eligible_pool_is_reported_not_silently_dropped():
    positives = pd.DataFrame({
        "human_id": ["H1"], "mouse_id": ["M1"], "tissue": ["liver"],
    })
    with pytest.raises(ValueError, match="негатив"):
        build_pospos_dataset(positives, progress=False)


def test_conflicting_donor_features_are_not_silently_selected():
    positives = pd.DataFrame({
        "human_id": ["H1", "H2", "H3"],
        "mouse_id": ["M1", "M1", "M2"],
        "tissue": ["liver"] * 3,
        "mouse_sequence": ["AA", "CC", "GG"],
    })
    with pytest.raises(ValueError, match="mouse"):
        build_pospos_dataset(positives, progress=False)


def test_pairwise_columns_require_explicit_recalculation():
    positives = pd.DataFrame({
        "human_id": ["H1", "H2"], "mouse_id": ["M1", "M2"],
        "tissue": ["liver"] * 2, "pair_similarity": [0.8, 0.9],
    })
    with pytest.raises(ValueError, match="pair_similarity"):
        build_pospos_dataset(positives, progress=False)
