# Eclare

<p align="center">
  <img src="img/eclare_arc.png" width="650">
</p>

<p align="center">
  <b>Evolutionary Contrastive Learning for Alignment of Regulatory Elements</b>
</p>

## О проекте

**Eclare** — исследовательский проект по обучению общих представлений для регуляторных элементов генома разных видов.

Основная идея — использовать эволюционно связанные энхансеры как положительные пары в контрастивном обучении. Последовательности человека и мыши кодируются DNA language model, после чего небольшая projection head обучается сближать представления функционально соответствующих энхансеров и разделять несвязанные пары.

В отличие от прямого сравнения нуклеотидных последовательностей, Eclare исследует возможность выравнивать регуляторные элементы в пространстве признаков, сохраняя их функциональное сходство даже при значительном расхождении последовательностей.

<p align="center">
  <i>sequence → DNA language model → shared regulatory embedding space</i>
</p>

## Статус

Проект находится на ранней исследовательской стадии. Сейчас формируется первый воспроизводимый пайплайн для построения межвидового датасета энхансеров и обучения базовой контрастивной модели.

## Первый DNA baseline

[notebooks/train_baseline.ipynb](notebooks/train_baseline.ipynb) — пошаговый
DNABERT-2 (заморожен) → общая MLP → multi-positive cross-species SupCon.
Короткие ячейки с русскими пояснениями, `.head()` и tqdm; параметры обучения
заданы отдельно, без проверок версий и служебного preflight.
Он подготовлен **без запуска**: обучение, метрики и график ещё не вычислены.
Отдельное окружение Python 3.10 описано в [requirements-baseline.txt](requirements-baseline.txt);
ячейки не устанавливают зависимости. DNABERT-2 использует зафиксированный remote
code (`trust_remote_code=True`), обычную PyTorch attention без Triton.

- [eclare/data/negatives.py](eclare/data/negatives.py): `build_pospos_dataset`
  добавляет один `pospos_shuffled_K1` на позитивную строку. Mouse выбирается
  равномерно среди уникальных IDs той же ткани; все известные позитивные
  партнёры human в этой ткани исключены. Переносятся все `mouse_*` поля.
  Нет GC/length matching или изменения DNA. Пустой пул/конфликтующие признаки
  вызывают ошибку, а не молчаливую потерю строк.
- [eclare/data/vista.py](eclare/data/vista.py): зафиксированный `locus.tsv.gz`
  из [vista-data](https://gitlab.com/egsb-mfgl/vista-data), commit
  `0ede19e42ffdcee88e250721a9198c7766627f8d`, SHA256-проверка и overlap ≥1 bp.
  Используются только нативные `assembly/coord`, не межвидовые liftOver-колонки.
- Validation — **VISTA-supported SCREEN retrieval (proxy)**: human ELS
  пересекает positive VISTA hg38-фрагмент; mouse truth — прямые SCREEN homologs.
  VISTA здесь не доказывает функциональную эквивалентность mouse-партнёра.
  Все VISTA-tested локусы, их homolog-компоненты, пересекающиеся интервалы
  и точные DNA-дубликаты исключены из train до sampling. Это явно отличается
  от исходной human-only CV/LOTO. Почти идентичные DNA не кластеризуются.
- Отдельные ячейки оценивают raw DNABERT-2 и обученную MLP на одинаковом
  фиксированном held-out mouse-пуле: recall@1/5/10/20, hit@k, MRR, random baseline.
  Matplotlib small multiples «до/после» сохраняются в PDF и PNG при запуске.
  При одном правильном mouse hit@k и recall@k совпадают; при one-to-many — нет.

Исходный вход — `data/processed/baseline_with_dna.parquet`; создание позитивов
остаётся в пользовательском `notebooks/create_pos_pairs.ipynb`.
Результаты, split, параметры и финальная голова сохраняются в новую папку
`data/experiments/baseline_<timestamp>/`, без перезаписи исходного parquet.
Обучается только DNA-голова, assay/tissue признаки в модель не подаются.
SupCon использует K1-кандидатов **и in-batch negatives**, поэтому его эффективный
знаменатель шире соотношения 1:1 в сохранённой таблице.

## SCREEN metadata

`eclare.data.screen` загружает ELS-пары и biosample-specific метаданные через
GraphQL. `load_orthologous_els` использует пятую колонку SCREEN BED (`EH38E…` /
`EM10E…`); связи one-to-many сохраняются, точные дубликаты пар удаляются.
`fetch_ccres` возвращает сырые записи, `parse_zscores` выбирает колонки,
`fetch_ccre_metadata` сразу возвращает DataFrame для одного вида,
`fetch_human_mouse_metadata` возвращает `pairs, human_meta, mouse_meta`.

```python
from eclare.data.screen import fetch_human_mouse_metadata, enrich_ccre_metadata

fields = ["tissue", "sample_type", "life_stage", "assay", "zscore"]
pairs, human_meta, mouse_meta = fetch_human_mouse_metadata(
    "hg38-mm10-Homologous.tsv", "GRCh38-cCREs.ELS.bed", "mm10-cCREs.ELS.bed",
    fields=fields, batch_size=200,
)
pairs.to_parquet("orthologous_els.parquet", index=False)
human_meta.to_parquet("human_metadata.parquet", index=False)
mouse_meta.to_parquet("mouse_metadata.parquet", index=False)
```

Поддерживаются `ccre_id`, `ccre_group`, `biosample`, `display_name`, `tissue`,
`sample_type`, `life_stage`, `assay`, `zscore`, `tf`, `experiment_accession`,
`file_accession`. `ccre_id` добавляется автоматически; без `fields` возвращаются
все эти колонки. `tissue` соответствует SCREEN `ontology`.
`include_biosample_details=true` используется только при запросе `display_name`,
`tissue`, `sample_type` или `life_stage`. Позиции полей в ответе без details
сохраняются: вместо этих значений приходят `null`
([формат SCREEN](https://github.com/weng-lab/SCREEN2.0/blob/b256bde1dcfba45aad50fc255459d911be75e63e/screen2.0/src/common/lib/zscores.ts#L3-L21)).

Позже можно добавить поля или новые IDs, не заменяя существующие значения:

```python
import pandas as pd

human_meta = pd.read_parquet("human_metadata.parquet")
human_meta = enrich_ccre_metadata(
    human_meta, pairs["human_id"].unique(), assembly="grch38",
    fields=[*fields, "display_name"],
)
human_meta.to_parquet("human_metadata.parquet", index=False)
```

Покрытие определяется по колонкам и обработанным IDs, а не по `NaN`.
`DataFrame.attrs["screen"]` хранит покрытие по полям, включая cCRE без Z-score
строк, assembly и ключи измерений. Эти атрибуты сохраняются при pandas/pyarrow
Parquet round-trip; для дополнения передавайте исходную таблицу без перестановки
или фильтрации строк. Таблицы без атрибутов тоже принимаются: колонки считаются
полученными для присутствующих IDs, а строки должны однозначно сопоставляться
по видимым ключам измерений. Неоднозначное сопоставление или изменение набора
измерений между запросами приводит к явной ошибке.

Установка зависимостей: `python -m pip install -r requirements.txt`.
Короткий E2E-пример с минимальным набором полей и экспортом Parquet:
[notebooks/screen_metadata.ipynb](notebooks/screen_metadata.ipynb).
Ключ задаётся через `SCREEN_API_KEY` или аргумент `api_key`; инструкции —
[SCREEN API Console](https://console.wenglab.org/docs/getting-started/auth).

Батч по умолчанию — 200 IDs, timeout — 60 секунд, до трёх повторных попыток
при сетевых ошибках/HTTP 408, 429, 5xx с exponential backoff.
Для продолжения после сбоя повторите вызов: успешные ответы уже сохранены в
`.screen_cache/` и используются даже при изменении размера батча или набора IDs.
Старые checkpoint’ы читаются без миграции. Ответ с biosample details подходит и
для запроса без details, но не наоборот; при дополнении поля могут быть извлечены
из кеша вообще без сети. Кеш учитывает endpoint, assembly и запрос.
`cache_dir=None` отключает его. Bulk Z-score matrices не используются.
