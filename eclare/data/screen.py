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
    "ccre_id", "ccre_group", "biosample", "display_name", "ontology",
    "sample_type", "life_stage", "assay", "zscore", "tf",
    "experiment_accession", "file_accession",
]


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
    endpoint=SCREEN_ENDPOINT, api_key=None,
):
    """Fetch raw cCRE records in batches; reuse atomic JSON checkpoints.

    assembly is grch38 (GRCh38 accepted) or mm10. retries counts additional
    attempts after the first request. Only network errors and HTTP 408/429/5xx
    are retried, with exponential backoff (and numeric Retry-After if present).
    GraphQL errors or missing accessions fail explicitly, without checkpointing.
    Set SCREEN_API_KEY or pass api_key for authenticated SCREEN access.
    Repeating a call with the same IDs/batch_size reuses completed batches;
    cache keys also include the endpoint and query. cache_dir=None disables it.
    """
    assembly = assembly.lower()
    if assembly not in {"grch38", "mm10"}:
        raise ValueError("assembly must be grch38 or mm10")
    if batch_size < 1 or timeout <= 0 or retries < 0 or backoff < 0:
        raise ValueError("batch_size/timeout must be positive; retries/backoff nonnegative")
    accessions = sorted(set(accessions))
    prefix = "EH38E" if assembly == "grch38" else "EM10E"
    if any(re.fullmatch(rf"{prefix}\d+", accession) is None for accession in accessions):
        raise ValueError(f"{assembly} requires {prefix} accessions")
    cache_path = Path(cache_dir) / assembly if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
    api_key = api_key or os.environ.get("SCREEN_API_KEY")
    records = []

    with requests.Session() as session, tqdm(
        total=len(accessions), desc=f"SCREEN {assembly}", unit="cCRE", disable=not progress,
    ) as bar:
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        for start in range(0, len(accessions), batch_size):
            batch = accessions[start:start + batch_size]
            checkpoint = None
            if cache_path is not None:
                fingerprint = json.dumps([endpoint, QUERY, assembly, batch]).encode()
                checkpoint = cache_path / (hashlib.sha256(fingerprint).hexdigest() + ".json")
            cached = checkpoint is not None and checkpoint.exists()
            if cached:
                batch_records = json.loads(checkpoint.read_text(encoding="utf-8"))
            else:
                for attempt in range(retries + 1):
                    response = None
                    try:
                        response = session.post(
                            endpoint, json={"query": QUERY, "variables": {
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
            if checkpoint is not None and not cached:
                temporary = checkpoint.with_suffix(".tmp")
                temporary.write_text(json.dumps(batch_records), encoding="utf-8")
                temporary.replace(checkpoint)
            records.extend(batch_records)
            bar.update(len(batch))
    return records


def parse_zscores(ccres):
    """One row per cCRE/experiment/file/assay; retain all Z-scores.

    zscores contains ten positional fields as documented in this project.
    Nine-field rows (older SCREEN responses without tf) use tf=None.
    cCREs with null/empty zscores have no metadata rows.
    """
    rows = []
    for ccre in ccres:
        for score in ccre["zscores"] or []:
            if not isinstance(score, (list, tuple)) or len(score) not in {9, 10}:
                raise ValueError(f"{ccre['accession']}: expected 9 or 10 Z-score fields")
            rows.append([
                ccre["accession"], ccre["ccre_group"], score[3], score[4], score[5],
                score[6], score[7], score[2], score[8],
                score[9] if len(score) == 10 else None, score[0], score[1],
            ])
    metadata = pd.DataFrame(rows, columns=META_COLUMNS)
    metadata["zscore"] = pd.to_numeric(metadata["zscore"], errors="raise")
    return metadata


def fetch_human_mouse_metadata(
    homologs_path, human_els_path, mouse_els_path, batch_size=200, *,
    timeout=60, retries=3, backoff=1.0, cache_dir=".screen_cache",
    progress=True, endpoint=SCREEN_ENDPOINT, api_key=None,
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
    )
    species = [("human_id", "grch38"), ("mouse_id", "mm10")]
    ids = {column: sorted(pairs[column].unique()) for column, _ in species}
    previews = {}
    # Check both assemblies before committing to either full download.
    for column, assembly in species:
        previews[column] = fetch_ccres(ids[column][:3], assembly, **options)
        parse_zscores(previews[column])
    metadata = []
    for column, assembly in species:
        remaining = fetch_ccres(ids[column][3:], assembly, **options)
        metadata.append(parse_zscores(previews[column] + remaining))
    return pairs, metadata[0], metadata[1]
