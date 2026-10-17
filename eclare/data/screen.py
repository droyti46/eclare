"""
Eclare — Evolutionary Contrastive Learning for Alignment of Regulatory Elements.

Evaluation utilities for cross-species regulatory element alignment.

Copyright (c) 2026 Eclare contributors.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

SCREEN_ENDPOINT = "https://screen.api.wenglab.org/graphql"
QUERY = """
query GetcCREZScoresQuery($assembly: String!, $accessions: [String]!) {
  getcCREZScoresQuery(
    assembly: $assembly
    accession: $accessions
    nearbygeneslimit: 1
    include_biosample_details: true
  ) {
    accession
    ccre_group
    zscores
  }
}
"""
META_COLUMNS = [
    "ccre_id", "ccre_group", "biosample", "display_name", "tissue",
    "sample_type", "life_stage", "assay", "zscore", "tf",
    "experiment_accession", "file_accession",
]

DETAIL_FIELDS = {"display_name", "tissue", "sample_type", "life_stage"}
ROW_KEYS = ["ccre_id", "experiment_accession", "file_accession", "assay", "biosample"]


def _fields(fields):
    columns = list(dict.fromkeys(["ccre_id", *(META_COLUMNS if fields is None else fields)]))
    unknown = set(columns) - set(META_COLUMNS)
    if unknown:
        raise ValueError(f"Unsupported fields: {sorted(unknown)}")
    return columns


def _accessions(accessions, assembly):
    assembly = assembly.lower()
    if assembly not in {"grch38", "mm10"}:
        raise ValueError("assembly must be grch38 or mm10")
    accessions = sorted(set(accessions))
    prefix = "EH38E" if assembly == "grch38" else "EM10E"
    if any(re.fullmatch(rf"{prefix}\d+", accession) is None for accession in accessions):
        raise ValueError(f"{assembly} requires {prefix} accessions")
    return accessions, assembly


def load_orthologous_els(homologs_path, human_els_path, mouse_els_path):
    """Filter headerless human/mouse accession pairs by both ELS BEDs.

    SCREEN BEDs have the cCRE accession in column 5 (column 4 is rDHS).
    Exact duplicate pairs are removed; one-to-many mappings are retained.
    Coordinates are not used or converted.
    """
    try:
        pairs = pd.read_csv(
            homologs_path, sep="\t", header=None, usecols=[0, 1],
            names=["human_id", "mouse_id"], dtype=str, comment="#",
        )
    except pd.errors.EmptyDataError:
        pairs = pd.DataFrame(columns=["human_id", "mouse_id"])

    for column, path, prefix in (
        ("human_id", human_els_path, "EH38E"),
        ("mouse_id", mouse_els_path, "EM10E"),
    ):
        pattern = rf"{prefix}\d+"
        if not pairs[column].str.fullmatch(pattern, na=False).all():
            raise ValueError(f"{homologs_path}: {column} must contain {prefix} accessions")
        try:
            ids = pd.read_csv(
                path, sep="\t", header=None, usecols=[4], dtype=str, comment="#",
            ).iloc[:, 0]
        except pd.errors.EmptyDataError:
            ids = pd.Series(dtype=str)
        if not ids.str.fullmatch(pattern, na=False).all():
            raise ValueError(f"{path}: BED column 5 must contain {prefix} accessions")
        pairs = pairs.loc[pairs[column].isin(ids)]

    return pairs.drop_duplicates().reset_index(drop=True)


def fetch_ccres(
    accessions, assembly, batch_size=200, *, timeout=60, retries=3,
    backoff=1.0, cache_dir=".screen_cache", progress=True,
    endpoint=SCREEN_ENDPOINT, api_key=None, fields=None,
):
    """Fetch raw cCRE records in batches; reuse atomic JSON checkpoints.

    assembly is grch38 (GRCh38 accepted) or mm10. retries counts additional
    attempts after the first request. Only network errors and HTTP 408/429/5xx
    are retried, with exponential backoff (and numeric Retry-After if present).
    GraphQL errors or missing accessions fail explicitly, without checkpointing.
    Set SCREEN_API_KEY or pass api_key for authenticated SCREEN access.
    Completed IDs are reused even if batch_size or the requested ID set changes.
    Detailed checkpoints also satisfy requests without biosample details; compact
    checkpoints do not satisfy detail requests. Old raw-list checkpoints are read
    using their original hash. cache_dir=None disables checkpointing.
    """
    columns = _fields(fields)
    details = bool(DETAIL_FIELDS.intersection(columns))
    query = QUERY if details else QUERY.replace(
        "include_biosample_details: true", "include_biosample_details: false",
    )
    accessions, assembly = _accessions(accessions, assembly)
    if batch_size < 1 or timeout <= 0 or retries < 0 or backoff < 0:
        raise ValueError("batch_size/timeout must be positive; retries/backoff nonnegative")
    cache_path = Path(cache_dir) / assembly if cache_dir is not None else None
    by_id = {}
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        wanted = set(accessions)
        for path in sorted(cache_path.glob("*.json")):
            if wanted.issubset(by_id):
                break
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, list):
                batch_records, saved_query = saved, QUERY
            else:
                if saved["endpoint"] != endpoint or saved["assembly"] != assembly:
                    continue
                batch_records, saved_query = saved["records"], saved["query"]
            if saved_query not in ({QUERY} if details else {QUERY, query}):
                continue
            saved_ids = sorted(row["accession"] for row in batch_records)
            fingerprint = json.dumps([endpoint, saved_query, assembly, saved_ids]).encode()
            if path.stem != hashlib.sha256(fingerprint).hexdigest():
                continue
            by_id.update((row["accession"], row) for row in batch_records if row["accession"] in wanted)
    pending = [id_ for id_ in accessions if id_ not in by_id]
    api_key = api_key or os.environ.get("SCREEN_API_KEY")

    with requests.Session() as session, tqdm(
        total=len(accessions), desc=f"SCREEN {assembly}", unit="cCRE", disable=not progress,
    ) as bar:
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        bar.update(len(accessions) - len(pending))
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            checkpoint = None
            if cache_path is not None:
                fingerprint = json.dumps([endpoint, query, assembly, batch]).encode()
                checkpoint = cache_path / (hashlib.sha256(fingerprint).hexdigest() + ".json")
            for attempt in range(retries + 1):
                response = None
                try:
                    response = session.post(
                        endpoint, json={"query": query, "variables": {
                            "assembly": assembly, "accessions": batch,
                        }}, timeout=timeout,
                    )
                    response.raise_for_status()
                except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
                    if isinstance(error, requests.HTTPError):
                        status = response.status_code
                        if status in {401, 403}:
                            raise RuntimeError(
                                f"SCREEN HTTP {status}: set SCREEN_API_KEY or pass api_key; "
                                "see https://console.wenglab.org/docs/getting-started/auth"
                            ) from error
                        if status not in {408, 429} and status < 500:
                            raise
                    if attempt == retries:
                        raise
                    delay = backoff * 2 ** attempt
                    if response is not None:
                        retry_after = response.headers.get("Retry-After", "")
                        if retry_after.isdigit():
                            delay = max(delay, int(retry_after))
                    time.sleep(delay)
                    continue

                payload = response.json()
                if payload.get("errors"):
                    raise RuntimeError(f"SCREEN GraphQL errors: {payload['errors']}")
                batch_records = (payload.get("data") or {}).get("getcCREZScoresQuery")
                break

            if not isinstance(batch_records, list):
                raise RuntimeError(f"SCREEN {assembly}: expected a list of cCRE records")
            returned = [record["accession"] for record in batch_records]
            if set(returned) != set(batch) or len(returned) != len(batch):
                missing = sorted(set(batch) - set(returned))
                unexpected = sorted(set(returned) - set(batch))
                raise RuntimeError(
                    f"SCREEN {assembly}: incomplete/duplicate response; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            if checkpoint is not None:
                temporary = checkpoint.with_suffix(".tmp")
                temporary.write_text(json.dumps({
                    "endpoint": endpoint, "assembly": assembly, "query": query,
                    "records": batch_records,
                }), encoding="utf-8")
                temporary.replace(checkpoint)
            by_id.update((row["accession"], row) for row in batch_records)
            bar.update(len(batch))
    return [by_id[id_] for id_ in accessions]


def parse_zscores(ccres, fields=None):
    """Return requested columns (plus ccre_id), without filtering measurements.

    tissue is SCREEN ontology. SCREEN keeps positions 4..7 as null placeholders
    without biosample details; the score remains at position 8. Older nine-field
    rows lack tf. Empty/null zscores produce no rows, but the
    processed accession is retained in attrs['screen']['coverage']. The attrs
    also retain measurement keys for later enrichment of projected tables.
    """
    columns = _fields(fields)
    rows = []
    processed = []
    for ccre in ccres:
        processed.append(ccre["accession"])
        for score in ccre["zscores"] or []:
            if not isinstance(score, (list, tuple)) or len(score) not in {9, 10}:
                raise ValueError(f"{ccre['accession']}: expected 9 or 10 Z-score fields")
            rows.append([
                ccre["accession"], ccre["ccre_group"], score[3], score[4], score[5],
                score[6], score[7], score[2], score[8],
                score[9] if len(score) == 10 else None, score[0], score[1],
            ])
    metadata = pd.DataFrame(rows, columns=META_COLUMNS)
    if "zscore" in columns:
        metadata["zscore"] = pd.to_numeric(metadata["zscore"], errors="raise")
    keys = metadata[ROW_KEYS].values.tolist()
    metadata = metadata[columns]
    metadata.attrs["screen"] = {
        "coverage": {column: sorted(set(processed)) for column in columns},
        "row_keys": keys,
    }
    return metadata


def fetch_ccre_metadata(accessions, assembly, fields=None, batch_size=200, **options):
    """Fetch selected metadata; transport options are those of fetch_ccres."""
    columns = _fields(fields)
    accessions, assembly = _accessions(accessions, assembly)
    metadata = parse_zscores(fetch_ccres(
        accessions, assembly, batch_size=batch_size, fields=columns, **options,
    ), fields=columns)
    metadata.attrs["screen"]["assembly"] = assembly
    return metadata


def enrich_ccre_metadata(metadata, accessions, assembly, fields, batch_size=200, **options):
    """Add missing fields/IDs without replacing existing values, including NaN.

    Coverage and measurement keys in DataFrame.attrs survive pandas/pyarrow
    Parquet round trips. Without attrs, existing columns count as processed for
    all IDs present in the table; shared columns must uniquely identify rows.
    Pass the unfiltered table returned by these functions to retain coverage
    for cCREs with no scores. The input DataFrame is not modified.
    """
    columns = _fields(fields)
    accessions, assembly = _accessions(accessions, assembly)
    if "ccre_id" not in metadata:
        raise ValueError("metadata must contain ccre_id")
    state = metadata.attrs.get("screen", {})
    if state.get("assembly", assembly) != assembly:
        raise ValueError("metadata assembly does not match assembly")
    _accessions(metadata["ccre_id"], assembly)
    present = sorted(set(metadata["ccre_id"]))
    coverage = {
        column: set(state.get("coverage", {}).get(column, present))
        for column in metadata.columns
    }
    groups = {}
    for id_ in accessions:
        missing = tuple(column for column in columns if id_ not in coverage.get(column, set()))
        if missing:
            groups.setdefault(missing, []).append(id_)
    result = metadata.copy(deep=True)
    row_keys = state.get("row_keys")
    if row_keys is not None and (
        len(row_keys) != len(result) or [key[0] for key in row_keys] != result["ccre_id"].tolist()
    ):
        raise ValueError("SCREEN row keys no longer match metadata; use the original unfiltered table")
    for missing, ids in groups.items():
        # Keys are retained separately so requested output columns stay minimal.
        extra = fetch_ccre_metadata(ids, assembly, fields=missing, batch_size=batch_size, **options)
        old_mask = result["ccre_id"].isin(ids)
        old = result.loc[old_mask]
        if not old.empty:
            if row_keys is not None:
                left = pd.DataFrame([key for key, keep in zip(row_keys, old_mask) if keep], columns=ROW_KEYS)
                right = pd.DataFrame(extra.attrs["screen"]["row_keys"], columns=ROW_KEYS)
                join_columns = ROW_KEYS
            else:
                # External tables can be enriched only when visible keys are unambiguous.
                common = [column for column in ROW_KEYS if column in old.columns]
                left = old[common].copy()
                right = pd.DataFrame(extra.attrs["screen"]["row_keys"], columns=ROW_KEYS)[common]
                join_columns = common
                if left.duplicated().any() or right.duplicated().any():
                    raise ValueError("Ambiguous metadata rows: retain SCREEN attrs or measurement key columns")
            left = left.reset_index(drop=True)
            right = right.reset_index(drop=True)
            left["_occurrence"] = left.groupby(join_columns, dropna=False).cumcount()
            right["_occurrence"] = right.groupby(join_columns, dropna=False).cumcount()
            right["_row"] = range(len(right))
            aligned = left.merge(right, on=join_columns + ["_occurrence"], how="left", validate="one_to_one")
            if aligned["_row"].isna().any() or len(extra.loc[extra["ccre_id"].isin(old["ccre_id"])]) != len(old):
                raise ValueError("SCREEN measurements changed; fetch a fresh table instead of mixing snapshots")
            positions = aligned["_row"].astype(int).tolist()
            for column in missing:
                if column not in result:
                    result[column] = pd.Series(index=result.index, dtype=extra[column].dtype)
                result.loc[old_mask, column] = extra.iloc[positions][column].to_numpy()
        new_mask = ~extra["ccre_id"].isin(result["ccre_id"])
        if extra.loc[new_mask, "ccre_id"].isin(coverage["ccre_id"]).any():
            raise ValueError("SCREEN measurements changed; fetch a fresh table instead of mixing snapshots")
        if new_mask.any():
            new = extra.loc[new_mask].copy()
            new.attrs = {}
            result.attrs = {}
            result = pd.concat([result, new], ignore_index=True)
            if row_keys is not None:
                row_keys = row_keys + [key for key, keep in zip(extra.attrs["screen"]["row_keys"], new_mask) if keep]
        for column in missing:
            coverage.setdefault(column, set()).update(ids)
    for column in columns:
        coverage.setdefault(column, set())
        if column not in result:
            result[column] = pd.Series(index=result.index, dtype=object)
    result.attrs["screen"] = {
        "assembly": assembly,
        "coverage": {column: sorted(ids) for column, ids in coverage.items()},
    }
    if row_keys is not None:
        result.attrs["screen"]["row_keys"] = row_keys
    return result


def fetch_human_mouse_metadata(
    homologs_path, human_els_path, mouse_els_path, batch_size=200, *,
    timeout=60, retries=3, backoff=1.0, cache_dir=".screen_cache",
    progress=True, endpoint=SCREEN_ENDPOINT, api_key=None, fields=None,
):
    """Return (pairs, human_meta, mouse_meta) for ELS homologous pairs.

    Unique IDs are fetched once per species. By default, each species first
    requests at most three IDs as a small API check, then loads all remaining
    IDs. The check uses the same persistent cache, so it is reused on restart.
    No biosample, assay, or Z-score threshold filtering is performed.
    """
    pairs = load_orthologous_els(homologs_path, human_els_path, mouse_els_path)
    options = dict(
        batch_size=batch_size, timeout=timeout, retries=retries, backoff=backoff,
        cache_dir=cache_dir, progress=progress, endpoint=endpoint, api_key=api_key,
        fields=_fields(fields),
    )
    species = [("human_id", "grch38"), ("mouse_id", "mm10")]
    ids = {column: sorted(pairs[column].unique()) for column, _ in species}
    previews = {}
    # Check both assemblies before committing to either full download.
    for column, assembly in species:
        previews[column] = fetch_ccres(ids[column][:3], assembly, **options)
        parse_zscores(previews[column], fields=fields)
    metadata = []
    for column, assembly in species:
        remaining = fetch_ccres(ids[column][3:], assembly, **options)
        frame = parse_zscores(previews[column] + remaining, fields=fields)
        frame.attrs["screen"]["assembly"] = assembly
        metadata.append(frame)
    return pairs, metadata[0], metadata[1]
