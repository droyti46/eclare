"""
Eclare — Evolutionary Contrastive Learning for Alignment of Regulatory Elements.

Evaluation utilities for cross-species regulatory element alignment.

Copyright (c) 2026 Eclare contributors.
"""

import gzip
import hashlib

import pandas as pd

import eclare.data.vista as vista_module
from eclare.data.vista import load_vista, overlap_vista


def test_native_vista_coordinates_ignore_cross_species_liftover(tmp_path, monkeypatch):
    text = (
        "vista_id\tcuration_status\ttissue\tassembly\tcoord\tcoordinate_hg38\tcoordinate_mm10\n"
        "hs1\tpositive\tlb\thg38\tchr1:11-20\tchr1:11-20\tchr2:101-110\n"
        "mm1\tnegative\t\tmm10\tchr3:1-9\tchr4:201-209\tchr3:1-9\n"
    )
    payload = gzip.compress(text.encode())
    (tmp_path / "locus.tsv.gz").write_bytes(payload)
    monkeypatch.setattr(vista_module, "VISTA_SHA256", hashlib.sha256(payload).hexdigest())
    result = load_vista(tmp_path, progress=False)
    assert result[["vista_id", "species", "chrom", "start", "end"]].values.tolist() == [
        ["hs1", "human", "chr1", 10, 20], ["mm1", "mouse", "chr3", 0, 9],
    ]
    assert result.curation_status.tolist() == ["positive", "negative"]


def test_overlap_is_at_least_one_bp_not_touching_endpoints():
    elements = pd.DataFrame({
        "human_id": ["one_bp", "touch", "other_chrom", "inside"],
        "human_chrom": ["chr1", "chr1", "chr2", "chr1"],
        "human_start": [9, 20, 9, 12],
        "human_end": [11, 30, 11, 15],
    })
    vista = pd.DataFrame({
        "vista_id": ["hs1"], "chrom": ["chr1"], "start": [10], "end": [20],
    })
    hits = overlap_vista(elements, vista, species="human", progress=False)
    assert hits[["human_id", "vista_id", "overlap_bp"]].values.tolist() == [
        ["one_bp", "hs1", 1], ["inside", "hs1", 3],
    ]


def test_overlap_preserves_multiple_vista_hits_without_replicating_tissue_rows():
    elements = pd.DataFrame({
        "mouse_id": ["M1", "M1"], "mouse_chrom": ["chr1", "chr1"],
        "mouse_start": [10, 10], "mouse_end": [20, 20],
    })
    vista = pd.DataFrame({
        "vista_id": ["mm1", "mm2"], "chrom": ["chr1", "chr1"],
        "start": [19, 0], "end": [21, 11],
    })
    hits = overlap_vista(elements, vista, species="mouse", progress=False)
    assert sorted(hits.overlap_bp.tolist()) == [1, 1]
    assert set(hits.vista_id) == {"mm1", "mm2"}
