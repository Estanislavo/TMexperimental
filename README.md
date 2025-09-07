# TMExperimental

Эксперименты и сравнение тематических моделей (**gensim LDA / BigARTM / ContextualTop2Vec**) на корпусах **OCTIS**.

---

## Что это

**TMExperimental** — утилита для сравнения тематических моделей на стандартизованных корпусах **OCTIS**. Скрипт:

1. Загружает корпус (по имени OCTIS или по локальному пути в формате OCTIS);
2. Обучает последовательно:
   - **LDA (gensim)**,
   - **BigARTM**,
   - **ContextualTop2Vec** (можно отключить флагом);
3. Извлекает топ-слова тем;
4. Считает метрики (из пакета **OCTIS**);
5. Печатает отчёт и сохраняет CSV.

---

## Требования

- **Python**: 3.10 (рекомендуется).
- **OS**: Linux рекомендуется. На Windows/macOS возможно, но могут возникнуть проблемы.

---

## Установка

Используем **Poetry**.

1) Если Poetry ещё нет:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2) Клонируем репозиторий и переходим в него:
```bash
git clone <url> TMExperimental
cd TMExperimental
```

3) Создайте окружение на Python 3.10 и установите зависимости:
```bash
poetry env use python3.10
poetry install
```

> Если при первом запуске CT2V появится `ModuleNotFoundError: No module named 'torch'`,
> поставьте CPU-колёса PyTorch и повторите:
> ```bash
> poetry run pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
> poetry install
> ```

---

## Быстрый старт

Запуск на 20NewsGroup, 20 тем:
```bash
poetry run tmexp-benchmark --dataset 20NewsGroup --topics 20
```

Подвыборка 500 документов (для быстрого прогона):
```bash
poetry run tmexp-benchmark --dataset 20NewsGroup --subset 500 --topics 20
```

После выполнения появится CSV со сводкой в текущей папке (или в `--outdir`).

---

## Корпуса OCTIS

Параметр `--dataset` принимает:
- **Имя встроенного датасета OCTIS** (скачивается при первом использовании):
  `20NewsGroup`, `BBC_News`, `AGNews`, `M10`, `YahooAnswers`, `NewYorkTimes`,
  `StackOverflow`, `TweetSent140`, `BioChem` и др.
- **Путь к локальной папке** в формате OCTIS (с файлами `documents.txt`, `vocabulary.txt`, и т.п.).

---

## CLI: параметры

### Общие
- `--dataset <str>` — имя OCTIS-корпуса или путь к локальному OCTIS-датасету. По умолчанию: `20NewsGroup`.
- `--topics <int>` — число тем **K**. По умолчанию: `20`.
- `--topk <int>` — сколько топ-слов темы учитывать в метриках. По умолчанию: `10`.
- `--print-topn <int>` — сколько слов печатать на тему в консоли. По умолчанию: `10`.
- `--subset <int>` — использовать первые **N** документов (0 — весь корпус). По умолчанию: `0`.
- `--outdir <path>` — куда сохранить CSV. По умолчанию: `"."`.
- `--workers <int>` — число потоков для LDA/CT2V. По умолчанию: `2`.

### LDA (gensim)
- `--lda-passes <int>` — проходов по коллекции. По умолчанию: `10`.
- `--lda-iters <int>` — итераций внутреннего оптимизатора. По умолчанию: `50`.
- `--lda-chunksize <int>` — размер чанка. По умолчанию: `2000`.

### BigARTM
- `--artm-passes <int>` — проходов offline-обучения. По умолчанию: `10`.
- `--artm-procs <int>` — процессов BigARTM. По умолчанию: `4`.
- `--tmpdir <path>` — база для временных батчей (по умолчанию системный TMP).

### ContextualTop2Vec
- `--no-ct2v` — не запускать CT2V.
- `--ct2v-embed <str>` — имя эмбеддинг-модели (напр., `all-MiniLM-L6-v2`). По умолчанию: `all-MiniLM-L6-v2`.
- `--ct2v-batch <int>` — размер батча эмбеддингов. По умолчанию: `8`.
---

## Какие метрики считаются

### Всегда (по умолчанию)
- `coh_c_v` — когерентность `c_v` (OCTIS `Coherence(measure="c_v")`)
- `coh_c_npmi` — когерентность `c_npmi`
- `coh_u_mass` — когерентность `u_mass`
- `coh_c_uci` — когерентность `c_uci` (добавлено)
- `topic_diversity` — разнообразие тем (OCTIS `TopicDiversity`)
- `inv_rbo` — `InvertedRBO` (добавлено), параметр веса регулируется `--irbo-weight`

### При наличии word2vec/fastText (задан `--w2v-path`)
- `we_coh_pairwise` — `WECoherencePairwise`
- `we_coh_centroid` — `WECoherenceCentroid`
- `we_inv_rbo` — `WordEmbeddingsInvertedRBO`
- `we_inv_rbo_centroid` — `WordEmbeddingsInvertedRBOCentroid`

> Все метрики используют **только** реализацию из OCTIS. Никаких самописных метрик в проекте нет (это удобно для сравнения с Ангеловым)

---

Быстрая «переустановка окружения Poetry» и запуск:
```bash
poetry env remove --all
poetry env use python3.10
poetry install
poetry run tmexp-benchmark --dataset 20NewsGroup --topics 20 --ct2v-embed all-MiniLM-L6-v2
```

---

## Производительность

- **CPU**: увеличивайте `--workers` (LDA/CT2V) и `--artm-procs` (BigARTM).
---
