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

## SCREEN metadata

`eclare.data.screen` содержит четыре функции: `load_orthologous_els`, `fetch_ccres`,
`parse_zscores`, `fetch_human_mouse_metadata`. Последняя возвращает
`pairs, human_meta, mouse_meta`: пары ELS из homologous TSV и biosample-specific
метаданные, полученные через GraphQL. Используется пятая колонка SCREEN BED;
связи one-to-many сохраняются, точные дубликаты пар удаляются.

Установка зависимостей: `python -m pip install -r requirements.txt`.
Пошаговый пример с проверкой на трёх cCRE каждого вида и экспортом Parquet:
[notebooks/screen_metadata.ipynb](notebooks/screen_metadata.ipynb).
Ключ задаётся через `SCREEN_API_KEY` или аргумент `api_key`; инструкции —
[SCREEN API Console](https://console.wenglab.org/docs/getting-started/auth).

Батч по умолчанию — 200 IDs, timeout — 60 секунд, до трёх повторных попыток
при сетевых ошибках/HTTP 408, 429, 5xx с exponential backoff.
Для продолжения после сбоя повторите вызов с теми же входами и размером батча:
успешные ответы уже сохранены в `.screen_cache/`. Кеш можно отключить через
`cache_dir=None`. Bulk Z-score matrices не используются.
