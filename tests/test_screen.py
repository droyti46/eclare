"""
Eclare — Evolutionary Contrastive Learning for Alignment of Regulatory Elements.

Evaluation utilities for cross-species regulatory element alignment.

Copyright (c) 2026 Eclare contributors.
"""

import hashlib
import json
from unittest.mock import Mock

import eclare.data.screen as screen

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
    assert all(call.kwargs["timeout"] == 60 for call in post.call_args_list)
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


def test_chunked_response_retried_then_checkpointed(tmp_path, monkeypatch):
    expected = [record("EH38E1")]
    post = Mock(side_effect=[
        requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead"),
        response(expected),
    ])
    monkeypatch.setattr(requests.Session, "post", post)
    sleep = Mock()
    monkeypatch.setattr("eclare.data.screen.time.sleep", sleep)
    options = dict(cache_dir=tmp_path, retries=1, backoff=2, progress=False)

    assert fetch_ccres(["EH38E1"], "grch38", **options) == expected
    assert post.call_count == 2
    assert post.call_args_list[0] == post.call_args_list[1]
    sleep.assert_called_once_with(2)
    assert len(list(tmp_path.rglob("*.json"))) == 1
    post.reset_mock()
    assert fetch_ccres(["EH38E1"], "grch38", **options) == expected
    post.assert_not_called()


@pytest.mark.parametrize("retries", [0, 2])
def test_chunked_response_exhausts_retries_without_checkpoint(tmp_path, monkeypatch, retries):
    error = requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")
    post = Mock(side_effect=error)
    monkeypatch.setattr(requests.Session, "post", post)
    sleep = Mock()
    monkeypatch.setattr("eclare.data.screen.time.sleep", sleep)

    with pytest.raises(requests.exceptions.ChunkedEncodingError) as caught:
        fetch_ccres(
            ["EH38E1"], "grch38", cache_dir=tmp_path,
            retries=retries, backoff=2, progress=False,
        )
    assert caught.value is error
    assert post.call_count == retries + 1
    assert [call.args[0] for call in sleep.call_args_list] == [2 * 2 ** i for i in range(retries)]
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.tmp"))


def test_retry_after_and_exponential_backoff(tmp_path, monkeypatch):
    limited = response(status=429)
    limited.headers["Retry-After"] = "5"
    post = Mock(side_effect=[limited, response(status=503), response([record("EH38E1")])])
    monkeypatch.setattr(requests.Session, "post", post)
    sleep = Mock()
    monkeypatch.setattr("eclare.data.screen.time.sleep", sleep)
    rows = fetch_ccres(
        ["EH38E1"], "grch38", retries=2, backoff=2, timeout=12,
        fields=["assay"], cache_dir=tmp_path, progress=False,
    )
    assert len(rows) == 1
    assert [call.args[0] for call in sleep.call_args_list] == [5, 4]
    assert all(call.kwargs["timeout"] == 12 for call in post.call_args_list)


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


def test_selected_fields_and_real_nulls():
    meta = parse_zscores(
        [record("EH38E1"), record("EH38E2", [])],
        fields=["tissue", "assay", "zscore", "tissue"],
    )
    assert list(meta.columns) == ["ccre_id", "tissue", "assay", "zscore"]
    assert meta.values.tolist() == [["EH38E1", "liver", "H3K27ac", 2.5]]
    assert "EH38E2" in meta.attrs["screen"]["coverage"]["tissue"]
    with pytest.raises(ValueError, match="fields"):
        parse_zscores([], fields=["unknown"])


@pytest.mark.parametrize("with_tf", [False, True])
def test_compact_response_keeps_zscore_at_index_eight(with_tf):
    score = ["ENCSR1", "ENCFF1", "DNase", "liver_ENCDO1", None, None, None, None, -2.5]
    if with_tf:
        score.append("na")
    meta = parse_zscores([record("EH38E1", [score])], fields=["assay", "zscore", "tf"])
    assert list(meta.columns) == ["ccre_id", "assay", "zscore", "tf"]
    assert meta.iloc[0, :3].tolist() == ["EH38E1", "DNase", -2.5]
    assert meta.iloc[0]["tf"] == ("na" if with_tf else None)


def test_detail_flag_and_cache_upgrade(tmp_path, monkeypatch):
    post = Mock(return_value=response([record("EH38E1")]))
    monkeypatch.setattr(requests.Session, "post", post)
    options = dict(cache_dir=tmp_path, progress=False)
    fetch_ccres(["EH38E1"], "grch38", fields=["assay"], **options)
    assert "include_biosample_details: false" in post.call_args.kwargs["json"]["query"]
    fetch_ccres(["EH38E1"], "grch38", fields=["tissue"], **options)
    assert post.call_count == 2
    assert "include_biosample_details: true" in post.call_args.kwargs["json"]["query"]
    fetch_ccres(["EH38E1"], "grch38", fields=["zscore"], **options)
    assert post.call_count == 2


def test_legacy_batch_reused_for_subset_and_new_ids(tmp_path, monkeypatch):
    # Old checkpoints contain raw lists; their hash encodes endpoint/query/IDs.
    batch = ["EH38E1", "EH38E2"]
    fingerprint = json.dumps([screen.SCREEN_ENDPOINT, screen.QUERY, "grch38", batch]).encode()
    folder = tmp_path / "grch38"
    folder.mkdir()
    (folder / (hashlib.sha256(fingerprint).hexdigest() + ".json")).write_text(
        json.dumps([record(id_) for id_ in batch]), encoding="utf-8",
    )
    post = Mock(return_value=response([record("EH38E3")]))
    monkeypatch.setattr(requests.Session, "post", post)
    rows = fetch_ccres(
        ["EH38E2", "EH38E3"], "grch38", fields=["tissue"],
        batch_size=1, cache_dir=tmp_path, progress=False,
    )
    assert [row["accession"] for row in rows] == ["EH38E2", "EH38E3"]
    assert post.call_args.kwargs["json"]["variables"]["accessions"] == ["EH38E3"]
    assert post.call_count == 1


def test_enrich_parquet_preserves_rows_nulls_and_empty_ccres(tmp_path, monkeypatch):
    score = record("EH38E1")["zscores"][0]
    other = ["ENCSR2", "ENCFF2", "DNase", "brain_ENCDO2", "Brain",
             None, "tissue", "embryonic", "1.2", "na"]
    records = [record("EH38E1", [score, other]), record("EH38E2", [])]
    post = Mock(return_value=response(records))
    monkeypatch.setattr(requests.Session, "post", post)
    options = dict(cache_dir=None, progress=False)
    meta = screen.fetch_ccre_metadata(
        ["EH38E1", "EH38E2"], "grch38", fields=["tissue", "assay", "zscore"], **options,
    )
    path = tmp_path / "meta.parquet"
    meta.to_parquet(path, index=False)
    meta = pd.read_parquet(path)
    original = meta.copy(deep=True)
    post.reset_mock()
    same = screen.enrich_ccre_metadata(
        meta, ["EH38E1", "EH38E2"], "grch38", fields=["tissue", "zscore"], **options,
    )
    pd.testing.assert_frame_equal(same, original)
    post.assert_not_called()
    # The API may return the same measurements in a different order.
    post.return_value = response([record("EH38E1", [other, score]), records[1]])
    enriched = screen.enrich_ccre_metadata(
        meta, ["EH38E1", "EH38E2"], "grch38",
        fields=["tissue", "sample_type", "life_stage", "assay", "zscore"], **options,
    )
    pd.testing.assert_frame_equal(enriched[original.columns], original)
    assert enriched["life_stage"].tolist() == ["adult", "embryonic"]
    assert post.call_count == 1
    pd.testing.assert_frame_equal(meta, original)
    enriched.to_parquet(path, index=False)
    post.reset_mock()
    screen.enrich_ccre_metadata(
        pd.read_parquet(path), ["EH38E1", "EH38E2"], "grch38",
        fields=["sample_type", "life_stage"], **options,
    )
    post.assert_not_called()


def test_enrich_new_ids_tracks_partial_field_coverage(tmp_path, monkeypatch):
    def post(self, endpoint, **kwargs):
        return response([record(id_) for id_ in kwargs["json"]["variables"]["accessions"]])

    monkeypatch.setattr(requests.Session, "post", post)
    options = dict(cache_dir=None, progress=False)
    meta = screen.fetch_ccre_metadata(
        ["EH38E1"], "grch38", fields=["tissue", "life_stage"], **options,
    )
    meta = screen.enrich_ccre_metadata(
        meta, ["EH38E1", "EH38E2"], "grch38", fields=["tissue"], **options,
    )
    assert meta["ccre_id"].tolist() == ["EH38E1", "EH38E2"]
    assert pd.isna(meta.loc[1, "life_stage"])
    spy = Mock(return_value=response([record("EH38E2")]))
    monkeypatch.setattr(requests.Session, "post", spy)
    meta = screen.enrich_ccre_metadata(
        meta, ["EH38E1", "EH38E2"], "grch38", fields=["life_stage"], **options,
    )
    assert meta["life_stage"].tolist() == ["adult", "adult"]
    assert spy.call_args.kwargs["json"]["variables"]["accessions"] == ["EH38E2"]


def test_selected_fields_e2e(inputs, tmp_path, monkeypatch):
    def post(self, endpoint, **kwargs):
        assert "include_biosample_details: true" in kwargs["json"]["query"]
        return response([record(id_) for id_ in kwargs["json"]["variables"]["accessions"]])

    monkeypatch.setattr(requests.Session, "post", post)
    pairs, human, mouse = fetch_human_mouse_metadata(
        *inputs, fields=["tissue", "sample_type", "life_stage", "assay", "zscore"],
        batch_size=200, cache_dir=tmp_path / "cache", progress=False,
    )
    assert len(pairs) == 4
    for frame in [human, mouse]:
        assert list(frame.columns) == ["ccre_id", "tissue", "sample_type", "life_stage", "assay", "zscore"]
    spy = Mock()
    monkeypatch.setattr(requests.Session, "post", spy)
    enriched = screen.enrich_ccre_metadata(
        human, pairs["human_id"].unique(), "grch38",
        fields=["tissue", "display_name"], cache_dir=tmp_path / "cache", progress=False,
    )
    assert enriched["display_name"].tolist() == ["Liver"] * 3
    spy.assert_not_called()


def test_enrich_external_frame_uses_visible_measurement_keys(monkeypatch):
    first = record("EH38E1")["zscores"][0]
    second = ["ENCSR2", "ENCFF2", "DNase", "brain_ENCDO2", "Brain",
              "brain", "tissue", "embryonic", "1.2", "na"]
    meta = parse_zscores([record("EH38E1", [first, second])])
    meta = meta.drop(columns=["life_stage"])
    meta.attrs = {}
    monkeypatch.setattr(requests.Session, "post", Mock(return_value=response([
        record("EH38E1", [second, first]),
    ])))
    enriched = screen.enrich_ccre_metadata(
        meta, ["EH38E1"], "grch38", fields=["life_stage"], cache_dir=None, progress=False,
    )
    assert enriched["life_stage"].tolist() == ["adult", "embryonic"]
    pd.testing.assert_frame_equal(enriched[meta.columns], meta)


def test_enrich_rejects_ambiguous_external_rows_and_changed_measurements(monkeypatch):
    score = record("EH38E1")["zscores"][0]
    second = ["ENCSR2", "ENCFF2", "DNase", "brain_ENCDO2", "Brain",
              "brain", "tissue", "embryonic", "1.2", "na"]
    meta = parse_zscores([record("EH38E1", [score, second])], fields=["tissue"])
    post = Mock(return_value=response([record("EH38E1", [second, score])]))
    monkeypatch.setattr(requests.Session, "post", post)
    external = meta.copy()
    external.attrs = {}
    with pytest.raises(ValueError, match="Ambiguous"):
        screen.enrich_ccre_metadata(
            external, ["EH38E1"], "grch38", fields=["display_name"], cache_dir=None, progress=False,
        )
    post.return_value = response([record("EH38E1", [score])])
    with pytest.raises(ValueError, match="measurements changed"):
        screen.enrich_ccre_metadata(
            meta, ["EH38E1"], "grch38", fields=["display_name"], cache_dir=None, progress=False,
        )


def test_enrich_mixed_new_ids_and_missing_fields(tmp_path, monkeypatch):
    requests_seen = []

    def post(self, endpoint, **kwargs):
        ids = kwargs["json"]["variables"]["accessions"]
        requests_seen.append(ids)
        return response([record(id_) for id_ in ids])

    monkeypatch.setattr(requests.Session, "post", post)
    options = dict(cache_dir=tmp_path / "cache", progress=False)
    meta = screen.fetch_ccre_metadata(["EH38E1"], "grch38", fields=["tissue"], **options)
    enriched = screen.enrich_ccre_metadata(
        meta, ["EH38E1", "EH38E2"], "grch38", fields=["tissue", "life_stage"], **options,
    )
    assert enriched.values.tolist() == [
        ["EH38E1", "liver", "adult"], ["EH38E2", "liver", "adult"],
    ]
    assert requests_seen == [["EH38E1"], ["EH38E2"]]
    spy = Mock()
    monkeypatch.setattr(requests.Session, "post", spy)
    screen.enrich_ccre_metadata(
        enriched, ["EH38E1", "EH38E2"], "grch38", fields=["life_stage"], **options,
    )
    spy.assert_not_called()


def test_enrichment_resumes_detail_batches_after_interruption(tmp_path, monkeypatch):
    ids = ["EH38E1", "EH38E2"]
    compact = record("EH38E1")["zscores"][0]
    compact[4:8] = [None] * 4
    post = Mock(return_value=response([record(id_, [compact]) for id_ in ids]))
    monkeypatch.setattr(requests.Session, "post", post)
    options = dict(cache_dir=tmp_path, progress=False, retries=0)
    meta = screen.fetch_ccre_metadata(ids, "grch38", fields=["assay", "zscore"], **options)
    post.side_effect = [response([record("EH38E1")]), requests.Timeout("interrupted")]
    with pytest.raises(requests.Timeout):
        screen.enrich_ccre_metadata(meta, ids, "grch38", fields=["tissue"], batch_size=1, **options)
    assert "tissue" not in meta.columns
    post.reset_mock(side_effect=True)
    post.return_value = response([record("EH38E2")])
    enriched = screen.enrich_ccre_metadata(meta, ids, "grch38", fields=["tissue"], batch_size=1, **options)
    assert enriched["tissue"].tolist() == ["liver", "liver"]
    assert post.call_count == 1
    assert post.call_args.kwargs["json"]["variables"]["accessions"] == ["EH38E2"]


def test_enrich_rejects_new_measurements_for_previously_empty_ccre(tmp_path, monkeypatch):
    post = Mock(return_value=response([record("EH38E1", [])]))
    monkeypatch.setattr(requests.Session, "post", post)
    options = dict(cache_dir=None, progress=False)
    meta = screen.fetch_ccre_metadata(
        ["EH38E1"], "grch38", fields=["tissue", "zscore"], **options,
    )
    path = tmp_path / "empty.parquet"
    meta.to_parquet(path, index=False)
    post.return_value = response([record("EH38E1")])
    with pytest.raises(ValueError, match="measurements changed"):
        screen.enrich_ccre_metadata(
            pd.read_parquet(path), ["EH38E1"], "grch38", fields=["life_stage"], **options,
        )


def test_enrich_preserves_existing_numeric_dtype(monkeypatch):
    score = record("EH38E1")["zscores"][0]
    score[8] = 2
    post = Mock(return_value=response([record("EH38E1", [score])]))
    monkeypatch.setattr(requests.Session, "post", post)
    options = dict(cache_dir=None, progress=False)
    meta = screen.fetch_ccre_metadata(["EH38E1"], "grch38", fields=["zscore"], **options)
    enriched = screen.enrich_ccre_metadata(meta, ["EH38E1"], "grch38", fields=["tissue"], **options)
    pd.testing.assert_frame_equal(enriched[meta.columns], meta)


def test_empty_enrichment_does_not_mark_new_column_as_downloaded(monkeypatch):
    meta = parse_zscores([record("EH38E1")], fields=["tissue"])
    options = dict(cache_dir=None, progress=False)
    meta = screen.enrich_ccre_metadata(meta, [], "grch38", fields=["life_stage"], **options)
    assert pd.isna(meta.loc[0, "life_stage"])
    post = Mock(return_value=response([record("EH38E1")]))
    monkeypatch.setattr(requests.Session, "post", post)
    meta = screen.enrich_ccre_metadata(meta, ["EH38E1"], "grch38", fields=["life_stage"], **options)
    assert meta["life_stage"].tolist() == ["adult"]
    assert post.call_count == 1


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
