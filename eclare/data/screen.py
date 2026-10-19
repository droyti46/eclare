"""
Eclare — Evolutionary Contrastive Learning for Alignment of Regulatory Elements.

Evaluation utilities for cross-species regulatory element alignment.

Copyright (c) 2026 Eclare contributors.
"""

import hashlib
import json
import math
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
                except (
                    requests.ConnectionError, requests.Timeout, requests.HTTPError,
                    requests.exceptions.ChunkedEncodingError,
                ) as error:
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


# ---------------------------------------------------------------------------
# Baseline-friendly streaming tissue metadata
# ---------------------------------------------------------------------------

_TISSUE_DATASET_VERSION = 1
_SCORE_INDEX = {
    "experiment_accession": 0,
    "file_accession": 1,
    "assay": 2,
    "biosample": 3,
    "display_name": 4,
    "tissue": 5,
    "sample_type": 6,
    "life_stage": 7,
    "zscore": 8,
    "tf": 9,
}


def _raw_checkpoint_path(batch, assembly, query, endpoint, raw_cache_dir):
    """Return the legacy raw-JSON checkpoint path for one exact batch."""
    if raw_cache_dir is None:
        return None
    fingerprint = json.dumps([endpoint, query, assembly, batch]).encode()
    return Path(raw_cache_dir) / assembly / (hashlib.sha256(fingerprint).hexdigest() + ".json")


def _load_raw_checkpoint(path, *, batch, assembly, query, endpoint):
    """Load one exact legacy checkpoint without scanning/loading the whole cache."""
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    if isinstance(saved, list):
        records, saved_query = saved, QUERY
    else:
        if saved.get("endpoint") != endpoint or saved.get("assembly") != assembly:
            return None
        records, saved_query = saved.get("records"), saved.get("query")
    if saved_query != query or not isinstance(records, list):
        return None
    returned = [record.get("accession") for record in records]
    if len(returned) != len(batch) or set(returned) != set(batch):
        return None
    return records


def _request_ccre_batch(
    session, batch, assembly, *, query, endpoint, timeout, retries, backoff,
):
    """Fetch a single SCREEN batch with the retry policy used by fetch_ccres."""
    for attempt in range(retries + 1):
        response = None
        try:
            response = session.post(
                endpoint,
                json={"query": query, "variables": {
                    "assembly": assembly, "accessions": batch,
                }},
                timeout=timeout,
            )
            response.raise_for_status()
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.HTTPError,
            requests.exceptions.ChunkedEncodingError,
        ) as error:
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
        records = (payload.get("data") or {}).get("getcCREZScoresQuery")
        if not isinstance(records, list):
            raise RuntimeError(f"SCREEN {assembly}: expected a list of cCRE records")
        returned = [record.get("accession") for record in records]
        if len(returned) != len(batch) or set(returned) != set(batch):
            missing = sorted(set(batch) - set(returned))
            unexpected = sorted(set(returned) - set(batch))
            raise RuntimeError(
                f"SCREEN {assembly}: incomplete/duplicate response; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return records
    raise AssertionError("unreachable")


def project_tissue_zscores(ccres, *, assays=None):
    """Project raw SCREEN measurements to cCRE/tissue/assay/zscore rows.

    No aggregation is performed: every SCREEN measurement with a tissue label
    is retained as its own row.  The heavier biosample/sample-type/life-stage
    fields are deliberately omitted from this baseline layer, so replicate
    measurements may have identical ``ccre_id``, ``tissue`` and ``assay`` keys.

    ``assays`` can optionally restrict the retained assay names.  Missing tissue
    labels and missing/non-finite z-scores are skipped.  The returned z-score is
    stored as float32 and the DataFrame has no SCREEN attrs/row-key payload.
    """
    if assays is None:
        allowed_assays = None
    elif isinstance(assays, str):
        allowed_assays = {assays}
    else:
        allowed_assays = set(assays)

    rows = []
    for ccre in ccres:
        accession = ccre["accession"]
        for score in ccre.get("zscores") or []:
            if not isinstance(score, (list, tuple)) or len(score) not in {9, 10}:
                raise ValueError(f"{accession}: expected 9 or 10 Z-score fields")

            assay = score[_SCORE_INDEX["assay"]]
            tissue = score[_SCORE_INDEX["tissue"]]
            raw_zscore = score[_SCORE_INDEX["zscore"]]

            if allowed_assays is not None and assay not in allowed_assays:
                continue
            if tissue is None or tissue == "":
                continue
            if raw_zscore is None or raw_zscore == "":
                continue

            try:
                zscore = float(raw_zscore)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{accession}: non-numeric zscore {raw_zscore!r}"
                ) from error
            if not math.isfinite(zscore):
                continue

            rows.append((accession, tissue, assay, zscore))

    frame = pd.DataFrame.from_records(
        rows, columns=["ccre_id", "tissue", "assay", "zscore"],
    )
    if frame.empty:
        return pd.DataFrame({
            "ccre_id": pd.Series(dtype="string"),
            "tissue": pd.Series(dtype="string"),
            "assay": pd.Series(dtype="string"),
            "zscore": pd.Series(dtype="float32"),
        })
    frame["zscore"] = frame["zscore"].astype("float32")
    return frame

def _tissue_dataset_dir(
    output_dir, accessions, assembly, *, batch_size, preview_size, assays,
):
    accessions_sha256 = hashlib.sha256("\n".join(accessions).encode()).hexdigest()
    config = {
        "version": _TISSUE_DATASET_VERSION,
        "assembly": assembly,
        "n_accessions": len(accessions),
        "accessions_sha256": accessions_sha256,
        "batch_size": batch_size,
        "preview_size": preview_size,
        "assays": (
            None if assays is None
            else [assays] if isinstance(assays, str)
            else sorted(set(assays))
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return Path(output_dir) / f"{assembly}-{fingerprint}", config


def _write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def fetch_ccre_tissue_metadata(
    accessions,
    assembly,
    output_dir=".screen_tissue",
    batch_size=200,
    *,
    assays=None,
    preview_size=3,
    timeout=60,
    retries=3,
    backoff=1.0,
    raw_cache_dir=".screen_cache",
    progress=True,
    endpoint=SCREEN_ENDPOINT,
    api_key=None,
    delete_reused_raw=False,
):
    """Stream compact cCRE × tissue × assay metadata to a Parquet dataset.

    Unlike :func:`fetch_ccres`, this function never accumulates all raw SCREEN
    responses in RAM and never writes newly downloaded 100+ MB raw JSON batches.
    Each batch is projected immediately to ``ccre_id``, ``tissue``, ``assay``
    and ``zscore`` without aggregation, then checkpointed as a Parquet shard.
    Re-running the function resumes from those shards.

    Existing legacy ``.screen_cache`` JSON files are reused *one batch at a
    time* when their exact hashes match. Set ``delete_reused_raw=True`` only if
    you deliberately want to reclaim those raw-cache files after a compact shard
    has been safely written.

    The return value is the directory of the Parquet dataset. Reading the whole
    dataset with pandas is optional; for large runs prefer ``pyarrow.dataset``.
    """
    accessions, assembly = _accessions(accessions, assembly)
    if batch_size < 1 or timeout <= 0 or retries < 0 or backoff < 0:
        raise ValueError("batch_size/timeout must be positive; retries/backoff nonnegative")
    if preview_size < 0:
        raise ValueError("preview_size must be nonnegative")
    # Tissue is score[5], therefore biosample details must be requested.
    query = QUERY
    dataset_dir, config = _tissue_dataset_dir(
        output_dir,
        accessions,
        assembly,
        batch_size=batch_size,
        preview_size=preview_size,
        assays=assays,
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "_manifest.json"
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("config") != config:
            raise RuntimeError(f"Incompatible existing dataset manifest: {manifest_path}")
    else:
        _write_json_atomic(manifest_path, {
            "config": config,
            "columns": ["ccre_id", "tissue", "assay", "zscore"],
            "rows": 0,
            "completed_batches": 0,
        })

    # Preserve the old 3-ID preview + remaining batches layout so an interrupted
    # run can reuse the raw JSON cache already downloaded by the previous code.
    batches = []
    preview_size = min(preview_size, len(accessions))
    if preview_size:
        batches.append(accessions[:preview_size])
    remaining = accessions[preview_size:]
    batches.extend(
        remaining[start:start + batch_size]
        for start in range(0, len(remaining), batch_size)
    )

    api_key = api_key or os.environ.get("SCREEN_API_KEY")
    total_rows = 0
    completed_batches = 0

    with requests.Session() as session, tqdm(
        total=len(accessions), desc=f"SCREEN tissue {assembly}", unit="cCRE",
        disable=not progress,
    ) as bar:
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"

        for batch_index, batch in enumerate(batches):
            part = dataset_dir / f"part-{batch_index:05d}.parquet"
            if part.exists():
                # Parquet footer gives row count without materialising the shard.
                try:
                    import pyarrow.parquet as pq
                    total_rows += pq.ParquetFile(part).metadata.num_rows
                except ImportError:
                    total_rows += len(pd.read_parquet(part, columns=["ccre_id"]))
                completed_batches += 1
                bar.update(len(batch))
                continue

            raw_path = _raw_checkpoint_path(
                batch, assembly, query, endpoint, raw_cache_dir,
            )
            records = _load_raw_checkpoint(
                raw_path, batch=batch, assembly=assembly, query=query, endpoint=endpoint,
            )
            reused_raw = records is not None
            if records is None:
                records = _request_ccre_batch(
                    session,
                    batch,
                    assembly,
                    query=query,
                    endpoint=endpoint,
                    timeout=timeout,
                    retries=retries,
                    backoff=backoff,
                )

            compact = project_tissue_zscores(records, assays=assays)
            temporary = part.with_suffix(".parquet.tmp")
            compact.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(part)
            total_rows += len(compact)
            completed_batches += 1

            # Drop the huge raw response before the next request.
            del compact
            del records

            if reused_raw and delete_reused_raw and raw_path is not None:
                raw_path.unlink(missing_ok=True)

            _write_json_atomic(manifest_path, {
                "config": config,
                "columns": ["ccre_id", "tissue", "assay", "zscore"],
                "rows": total_rows,
                "completed_batches": completed_batches,
            })
            bar.update(len(batch))

    _write_json_atomic(manifest_path, {
        "config": config,
        "columns": ["ccre_id", "tissue", "assay", "zscore"],
        "rows": total_rows,
        "completed_batches": completed_batches,
    })
    return dataset_dir


def fetch_human_mouse_tissue_metadata(
    homologs_path,
    human_els_path,
    mouse_els_path,
    output_dir=".screen_tissue",
    batch_size=200,
    *,
    assays=None,
    max_pairs=None,
    max_human_ids=None,
    random_state=42,
    preview_size=3,
    timeout=60,
    retries=3,
    backoff=1.0,
    raw_cache_dir=".screen_cache",
    progress=True,
    endpoint=SCREEN_ENDPOINT,
    api_key=None,
    delete_reused_raw=False,
):
    """Build baseline tissue metadata for homologous human/mouse ELS pairs.

    For a fast baseline, ``max_pairs`` deterministically subsamples homologous
    pair rows. ``max_human_ids`` instead keeps pairs whose human cCRE belongs to
    the first N sorted human IDs; this is useful for warm-starting from an old
    interrupted cache whose batches were created in sorted-ID order. The two
    options are mutually exclusive. Set both to ``None`` for the full dataset.
    No tissue-level aggregation is performed; every retained SCREEN measurement
    remains a row in the Parquet shards. The returned metadata objects are
    Parquet *dataset directories*, not giant in-memory DataFrames.

    Returns ``(pairs, human_dataset_dir, mouse_dataset_dir)``.
    """
    pairs = load_orthologous_els(homologs_path, human_els_path, mouse_els_path)
    if max_pairs is not None and max_human_ids is not None:
        raise ValueError("Use either max_pairs or max_human_ids, not both")
    if max_human_ids is not None:
        if max_human_ids < 1:
            raise ValueError("max_human_ids must be positive or None")
        full_human_ids = sorted(pairs["human_id"].unique())
        keep = set(full_human_ids[:max_human_ids])
        pairs = pairs.loc[pairs["human_id"].isin(keep)].reset_index(drop=True)
    elif max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive or None")
        if len(pairs) > max_pairs:
            pairs = pairs.sample(n=max_pairs, random_state=random_state).reset_index(drop=True)

    human_ids = sorted(pairs["human_id"].unique())
    mouse_ids = sorted(pairs["mouse_id"].unique())
    common = dict(
        output_dir=output_dir,
        batch_size=batch_size,
        assays=assays,
        preview_size=preview_size,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        raw_cache_dir=raw_cache_dir,
        progress=progress,
        endpoint=endpoint,
        api_key=api_key,
        delete_reused_raw=delete_reused_raw,
    )
    human_dir = fetch_ccre_tissue_metadata(human_ids, "grch38", **common)
    mouse_dir = fetch_ccre_tissue_metadata(mouse_ids, "mm10", **common)
    return pairs, human_dir, mouse_dir
