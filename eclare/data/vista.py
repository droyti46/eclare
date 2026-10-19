"""
Eclare — Evolutionary Contrastive Learning for Alignment of Regulatory Elements.

Evaluation utilities for cross-species regulatory element alignment.

Copyright (c) 2026 Eclare contributors.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

VISTA_COMMIT = "0ede19e42ffdcee88e250721a9198c7766627f8d"
VISTA_URL = f"https://gitlab.com/egsb-mfgl/vista-data/-/raw/{VISTA_COMMIT}/locus.tsv.gz"
VISTA_SHA256 = "afef62dc08b637761647dde6e9a2b960d28b041946dfb6c70512fa5b63310061"


def load_vista(directory, *, progress=True):
    """Загрузить зафиксированный locus.tsv.gz и вернуть нативные BED-интервалы.

    coord: 1-based closed → start-1, end (0-based half-open).
    Межвидовые coordinate_hg38/coordinate_mm10 намеренно не используются.
    В таблице остаются и positive, и negative: для исключения из train нужны все.
    """
    path = Path(directory) / "locus.tsv.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".gz.part")
        with requests.get(VISTA_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0)) or None
            with temporary.open("wb") as handle, tqdm(
                total=total, unit="B", unit_scale=True, desc="VISTA", disable=not progress,
            ) as bar:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
                    bar.update(len(chunk))
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != VISTA_SHA256:
            raise ValueError("VISTA: SHA256 загруженного файла не совпадает")
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != VISTA_SHA256:
        raise ValueError("VISTA: локальный файл отличается от зафиксированной версии")

    vista = pd.read_csv(path, sep="\t", dtype=str)
    vista = vista.loc[vista.assembly.isin(["hg38", "mm10"])].copy()
    coordinates = vista.coord.str.extract(r"^(chr[^:]+):(\d+)-(\d+)$")
    if coordinates.isna().any().any():
        raise ValueError("VISTA: неожиданный формат нативных координат")
    vista["chrom"] = coordinates[0]
    vista["start"] = coordinates[1].astype(int) - 1
    vista["end"] = coordinates[2].astype(int)
    if not ((vista.start >= 0) & (vista.end > vista.start)).all():
        raise ValueError("VISTA: некорректные интервалы")
    vista["species"] = vista.assembly.map({"hg38": "human", "mm10": "mouse"})
    return vista.reset_index(drop=True)


def overlap_vista(elements, vista, *, species, progress=True):
    """Все пересечения ≥1 bp; входные интервалы уже в одной сборке, BED-система."""
    columns = [f"{species}_{c}" for c in ("id", "chrom", "start", "end")]
    elements = elements[columns].drop_duplicates()
    hits = []
    for chrom, loci in tqdm(
        vista.groupby("chrom", sort=False), desc=f"VISTA overlap: {species}", disable=not progress,
    ):
        candidates = elements.loc[elements[columns[1]].eq(chrom)]
        for locus in loci.itertuples(index=False):
            selected = candidates.loc[
                (candidates[columns[2]] < locus.end) & (candidates[columns[3]] > locus.start)
            ]
            if selected.empty:
                continue
            block = selected[[columns[0]]].copy()
            block["vista_id"] = locus.vista_id
            block["overlap_bp"] = (
                np.minimum(selected[columns[3]].to_numpy(), locus.end)
                - np.maximum(selected[columns[2]].to_numpy(), locus.start)
            )
            hits.append(block)
    if not hits:
        return pd.DataFrame(columns=[columns[0], "vista_id", "overlap_bp"])
    return pd.concat(hits, ignore_index=True)
