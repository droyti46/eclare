"""
Eclare — Evolutionary Contrastive Learning for Alignment of Regulatory Elements.

Evaluation utilities for cross-species regulatory element alignment.

Copyright (c) 2026 Eclare contributors.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def build_pospos_dataset(positives, *, seed=42, progress=True):
    """Добавить один pospos_K1-негатив на каждую позитивную строку.

    Пул — уникальные mouse_id в той же tissue, исключения — все позитивные
    партнёры human_id в этой ткани. Повторы между строками разрешены.
    Все mouse_* признаки берутся у нового партнёра; human_* не меняются.
    GC/длина/хромосома не используются для matching. DNA не изменяется.
    Источник алгоритма: Enhancers/enhancers/notebooks/
    build_pairs_hard_negatives_enhancers.ipynb, build_pospos_shuffled(K=1).
    Порядок входных строк и первого появления mouse_id влияет на результат.

    В отличие от исходного ноутбука пустой пул вызывает ошибку, а не потерю
    строки. При конфликтующих mouse-признаках также требуется исправить вход.
    Split и запись файлов остаются за вызывающим кодом.
    """
    keys = ["human_id", "mouse_id", "tissue"]
    if positives.empty or positives[keys].isna().any().any():
        raise ValueError("Нужны непустые позитивы без пропусков в ID и tissue")
    extra = [c for c in positives if c != "tissue" and not c.startswith(("human_", "mouse_"))]
    if extra:
        raise ValueError(f"Парные/служебные колонки нужно убрать или пересчитать явно: {extra}")

    positives = positives.reset_index(drop=True)
    mouse_columns = [c for c in positives if c.startswith("mouse_")]
    donors = positives[["tissue", *mouse_columns]].drop_duplicates().reset_index(drop=True)
    if donors.duplicated(["mouse_id", "tissue"]).any():
        raise ValueError("Конфликтующие mouse_* признаки для одного (mouse_id, tissue)")

    pools, positions = {}, {}
    for tissue, rows in donors.groupby("tissue", sort=False):
        pools[tissue] = rows.index.to_numpy()
        positions[tissue] = {mouse: i for i, mouse in enumerate(rows.mouse_id)}
    forbidden = {
        (human, tissue): sorted({positions[tissue][mouse] for mouse in rows.mouse_id})
        for (human, tissue), rows in positives.groupby(["human_id", "tissue"], sort=False)
    }

    rng = np.random.default_rng(seed)
    donor_indices = []
    anchors = positives[["human_id", "tissue"]].itertuples(index=False, name=None)
    for human, tissue in tqdm(anchors, total=len(positives), desc="Негативы pospos_K1", disable=not progress):
        blocked = forbidden[human, tissue]
        available = len(pools[tissue]) - len(blocked)
        if available == 0:
            raise ValueError(f"Нет допустимого негатива для {human}, tissue={tissue}")
        # Индекс среди допустимых IDs: не создаём огромный список кандидатов для каждой строки.
        index = int(rng.choice(available, size=1, replace=False)[0])
        for excluded in blocked:
            if index >= excluded:
                index += 1
        donor_indices.append(pools[tissue][index])

    negatives = positives.copy()
    negatives[mouse_columns] = donors.loc[donor_indices, mouse_columns].reset_index(drop=True)
    result = pd.concat([
        positives.assign(label=1, negative_type="positive"),
        negatives.assign(label=0, negative_type="pospos_shuffled_K1"),
    ], ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)
