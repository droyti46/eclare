"""
Eclare — Evolutionary Contrastive Learning for Alignment of Regulatory Elements.

Evaluation utilities for cross-species regulatory element alignment.

Copyright (c) 2026 Eclare contributors.
"""

import json
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from eclare.data.screen import (
    META_COLUMNS, fetch_ccres, fetch_human_mouse_metadata,
    load_orthologous_els, parse_zscores,
)


def record(accession, scores=None):
    return {
        "accession": accession, "ccre_group": "dELS",
        "zscores": scores if scores is not None else [
            ["ENCSR1", "ENCFF1", "H3K27ac", "liver_ENCDO1", "Liver",
             "liver", "tissue", "adult", "2.5", "na"],
        ],
    }


def response(records=None, status=200, errors=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps({
        "errors": errors, "data": {"getcCREZScoresQuery": records},
    }).encode()
    return result


@pytest.fixture
def inputs(tmp_path):
    homologs, human, mouse = [tmp_path / name for name in ("pairs.tsv", "human.bed", "mouse.bed")]
    homologs.write_text(
        "EH38E1\tEM10E1\nEH38E1\tEM10E2\nEH38E1\tEM10E1\n"
        "EH38E2\tEM10E3\nEH38E3\tEM10E1\nEH38E4\tEM10E4\n"
        "EH38E5\tEM10E5\n", encoding="utf-8",
    )
    human.write_text("".join(
        f"chr1\t0\t10\tEH38D{i}\tEH38E{i}\tdELS\n" for i in (1, 2, 4, 5)
    ), encoding="utf-8")
    mouse.write_text("".join(
        f"chr1\t0\t10\tEM10D{i}\tEM10E{i}\tdELS\n" for i in (1, 2, 4, 5)
    ), encoding="utf-8")
    return homologs, human, mouse


def test_filter_and_zscore_fields(inputs):
    pairs = load_orthologous_els(*inputs)
    assert pairs.values.tolist() == [
        ["EH38E1", "EM10E1"], ["EH38E1", "EM10E2"],
        ["EH38E4", "EM10E4"], ["EH38E5", "EM10E5"],
    ]
    scores = record("EH38E1")["zscores"]
    metadata = parse_zscores([
        record("EH38E1", scores + [scores[0][:9]]), record("EH38E2", []),
        {"accession": "EH38E3", "ccre_group": "pELS", "zscores": None},
    ])
    assert list(metadata.columns) == META_COLUMNS
    assert metadata.iloc[0].to_dict() == dict(zip(META_COLUMNS, [
        "EH38E1", "dELS", "liver_ENCDO1", "Liver", "liver", "tissue", "adult",
        "H3K27ac", 2.5, "na", "ENCSR1", "ENCFF1",
    ]))
    assert metadata.shape == (2, 12)
    assert pd.isna(metadata.iloc[1]["tf"])
    with pytest.raises(ValueError, match="Z-score fields"):
        parse_zscores([record("EH38E1", [["bad"]])])


def test_resume_after_failure_and_retry(tmp_path, monkeypatch):
    post = Mock(side_effect=[response([record("EH38E1")]), requests.Timeout("interrupted")])
    monkeypatch.setattr(requests.Session, "post", post)
    options = dict(batch_size=1, cache_dir=tmp_path, retries=0, progress=False)
    with pytest.raises(requests.Timeout):
        fetch_ccres(["EH38E2", "EH38E1", "EH38E1"], "GRCh38", **options)
    assert len(list(tmp_path.rglob("*.json"))) == 1
    post.reset_mock(side_effect=True)
    post.side_effect = [response(status=429), response([record("EH38E2")])]
    sleep = Mock()
    monkeypatch.setattr("eclare.data.screen.time.sleep", sleep)
    options["retries"] = 1
    rows = fetch_ccres(["EH38E1", "EH38E2"], "grch38", **options)
    assert [row["accession"] for row in rows] == ["EH38E1", "EH38E2"]
    assert post.call_count == 2
    assert all(call.kwargs["json"]["variables"]["accessions"] == ["EH38E2"] for call in post.call_args_list)
    sleep.assert_called_once_with(1.0)
    post.reset_mock()
    fetch_ccres(["EH38E2", "EH38E1"], "grch38", **options)
    post.assert_not_called()


@pytest.mark.parametrize("reply,exception", [
    (response(errors=[{"message": "bad query"}]), RuntimeError),
    (response([]), RuntimeError),
    (response(status=403), RuntimeError),
    (response(status=400), requests.HTTPError),
])
def test_failures_not_cached_or_retried(tmp_path, monkeypatch, reply, exception):
    post = Mock(return_value=reply)
    monkeypatch.setattr(requests.Session, "post", post)
    with pytest.raises(exception):
        fetch_ccres(["EH38E1"], "grch38", cache_dir=tmp_path, progress=False)
    assert post.call_count == 1
    assert not list(tmp_path.rglob("*.json"))


def test_pipeline_and_parquet(inputs, tmp_path, monkeypatch):
    calls = []

    def post(self, endpoint, **kwargs):
        variables = kwargs["json"]["variables"]
        calls.append(variables)
        assert kwargs["timeout"] == 12
        assert self.headers["Authorization"] == "Bearer test-key"
        return response([record(accession) for accession in variables["accessions"]])

    monkeypatch.setattr(requests.Session, "post", post)
    result = fetch_human_mouse_metadata(
        *inputs, batch_size=2, timeout=12, cache_dir=tmp_path / "cache",
        progress=False, api_key="test-key",
    )
    pairs, human, mouse = result
    assert len(pairs) == 4 and len(human) == 3 and len(mouse) == 4
    assert [call["assembly"] for call in calls] == ["grch38", "grch38", "mm10", "mm10", "mm10"]
    for assembly, prefix in [("grch38", "EH38E"), ("mm10", "EM10E")]:
        requested = [id_ for call in calls if call["assembly"] == assembly for id_ in call["accessions"]]
        assert len(requested) == len(set(requested))
        assert all(id_.startswith(prefix) for id_ in requested)
    for name, frame in zip(("pairs", "human", "mouse"), result):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path))
    calls.clear()
    fetch_human_mouse_metadata(*inputs, batch_size=2, cache_dir=tmp_path / "cache", progress=False)
    assert calls == []


def test_empty_pairs_and_wrong_accessions(inputs, tmp_path, monkeypatch):
    homologs, human, mouse = inputs
    homologs.write_text("", encoding="utf-8")
    post = Mock()
    monkeypatch.setattr(requests.Session, "post", post)
    pairs, human_meta, mouse_meta = fetch_human_mouse_metadata(
        *inputs, cache_dir=None, progress=False,
    )
    assert pairs.empty and human_meta.empty and mouse_meta.empty
    assert list(human_meta.columns) == META_COLUMNS
    post.assert_not_called()
    homologs.write_text("EH38D1\tEM10E1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="EH38E"):
        load_orthologous_els(*inputs)
    homologs.write_text("EH38E1\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="EM10E"):
        load_orthologous_els(*inputs)
    with pytest.raises(ValueError, match="EM10E"):
        fetch_ccres(["EH38E1"], "mm10", cache_dir=tmp_path, progress=False)
