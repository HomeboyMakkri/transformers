# Анализ тональности SST-2 с Transformers

Учебный проект на семь дней: от токенизации и внутренних представлений
Transformer до сравнения классификаторов, анализа ошибок и локального Gradio
демо. Переиспользуемый код находится в `src/transformers_learning/`, а
ноутбуки объясняют этапы и вызывают этот код.

## Быстрый старт

Нужен Python 3.10 (версия проекта указана в `.python-version`). Перед
созданием виртуального окружения выберите этот интерпретатор. Зависимости
закреплены в `requirements.txt`; `requirements-dev.txt` добавляет Jupyter и
инструменты разработки.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Запустить локальное приложение из корня проекта:

```bash
python app.py
```

Откройте <http://127.0.0.1:7860>. Приложению нужен сохранённый артефакт
`fine_tuned_model/`; оно загружает его локально при создании интерфейса и не
обучает и не скачивает новую модель. Gradio sharing выключен.

Для ноутбуков запустите JupyterLab из корня проекта:

```bash
jupyter lab
```

Открывайте ноутбуки по порядку из следующего раздела. Ячейки интеграции могут
запускать обучение или инференс по полному holdout. Перед «Run All» проверьте
значения `RUN_FULL_TRAINING`, `RUN_DAY6_INTEGRATION`,
`RUN_DAY7_D7_01_INTEGRATION` и `RUN_DAY7_D7_04_INTEGRATION` в соответствующих
ноутбуках. В текущем checkout эти флаги включены; полные прогоны требуют
датасет и/или локальную модель и могут занять время.

## Порядок изучения

1. [День 1 — токенизация](notebooks/day1_tokenization.ipynb): `input_ids`,
   специальные токены и маски.
2. [День 2 — hidden states](notebooks/day2_hidden_states.ipynb): инференс и
   формы скрытых представлений.
3. [День 3 — attention](notebooks/day3_attention-matrix_visualization.ipynb):
   визуализация весов внимания и её ограничения.
4. [День 4 — frozen baseline](notebooks/day4_frozen_embedding_baseline.ipynb):
   признаки первого токена и Logistic Regression.
5. [День 5 — fine-tuning](notebooks/day5_fine_tuning.ipynb): обучение
   классификатора и мониторинг validation split.
6. [День 6 — сравнение](notebooks/day6_model_comparison.ipynb): обе модели на
   одном outer holdout.
7. [День 7 — ошибки и демо](notebooks/day7_error_analysis_and_demo.ipynb):
   FP/FN, отчёт и Gradio интерфейс.

План этапов и их границы описаны в [PLAN.md](PLAN.md), принятые решения — в
[SPEC.md](SPEC.md), исходные учебные задания — в `tasks/`.

## Данные и метки

Используется англоязычный датасет
[`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2). Его
метки бинарные: `0 = negative`, `1 = positive`. Нейтрального класса нет:
любой текст модель относит к одному из двух классов. Лицензия в карточке
датасета указана как `unknown`; соблюдайте условия источника и не
распространяйте скачанные данные.

Код использует размеченные строки SST-2 и воспроизводимый стратифицированный
outer split 80/20 (`random_state=42`). В Day 5 validation выделяется только из
outer train. Метрики Day 6 считаются на общем outer holdout; он не используется
для обучения, выбора модели или настройки параметров. Официальный SST-2 test
split не нужен проекту и не имеет публичных меток.

## Сохранённые результаты

В `comparison_results.txt` записано сравнение на одном и том же outer holdout
из 13 470 примеров:

| Модель | Macro F1 | Accuracy |
| --- | ---: | ---: |
| Fine-tuned DistilBERT | 0.942374 | 0.942984 |
| Frozen embeddings + Logistic Regression | 0.865108 | 0.866964 |

Разница macro F1 — `+0.077267` (7.73 процентного пункта). Day 5 validation
macro F1 (`0.941422`) — отдельная метрика мониторинга, не итоговая оценка.

`error_analysis.txt` содержит результаты того же outer holdout: 768 ошибок из
13 470 (`5.70%`), из них 298 false positives и 470 false negatives. Это
описательное измерение. Примеры, длина текста, attention и уверенность модели
не устанавливают причину ошибки.

Значения взяты из локально сохранённых файлов результатов. Они описывают этот
зафиксированный запуск; это не гарантия качества на других данных.

## Выходные файлы и как их получить

| Результат | Где создаётся | Как воспроизвести |
| --- | --- | --- |
| Attention heatmaps | День 3, `attention_layer*_head*.png` | Выполнить ячейки визуализации в [ноутбуке Дня 3](notebooks/day3_attention-matrix_visualization.ipynb) |
| `baseline_results.txt` | Frozen baseline и его run context | Выполнить baseline checkpoint в [ноутбуке Дня 4](notebooks/day4_frozen_embedding_baseline.ipynb) |
| `fine_tuned_model/`, `fine_tuned_results.txt` | Финальная модель эпохи 3 и validation-метрики | Выполнить обучение и сохранение в [ноутбуке Дня 5](notebooks/day5_fine_tuning.ipynb) |
| `comparison_results.txt`, `confusion_matrix_finetuned.png`, `confusion_matrix_baseline.png` | Парное сравнение на outer holdout | Выполнить integration checkpoint в [ноутбуке Дня 6](notebooks/day6_model_comparison.ipynb) |
| `error_analysis.txt` | Сводка, FP/FN и детерминированные примеры | Выполнить D7-01 и D7-03 integration cells в [ноутбуке Дня 7](notebooks/day7_error_analysis_and_demo.ipynb) |
| Локальное Gradio приложение | `app.py` | Из корня проекта выполнить `python app.py` |

Генерируемые данные и артефакты остаются вне Git. `.gitignore` исключает
`data/`, `fine_tuned_model/`, файлы результатов, confusion matrices, attention
plots, кэши и сериализованные веса. Для работы приложения локальный
`fine_tuned_model/` должен быть создан на Дне 5.

## Ограничения интерпретации

- Выход модели — бинарный. «Нейтральный» текст всё равно получает `negative`
  или `positive`.
- Softmax probabilities показывают относительную уверенность внутри этих двух
  классов; они не являются калиброванной вероятностью истинности.
- Ошибки на holdout и attention weights помогают исследовать поведение, но
  сами по себе не объясняют причин предсказания. Не используйте holdout для
  дообучения или выбора модели.
- Полный чистый запуск всех семи ноутбуков в рамках D7-06 не выполнялся.
  Указанные выше метрики и отчёт уже сохранены; при новом запуске сверяйте
  состояние integration cells и локальных артефактов.

## Проверки проекта

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/pyright
```
