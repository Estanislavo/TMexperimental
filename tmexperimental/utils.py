import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO


def pretty_print_topics(name: str, topics: List[List[str]], topn: int = 10) -> None:
    """Печатает первые N слов каждой темы — для быстрого просмотра."""
    print(f"\n--- {name}: темы (по {topn} слов) ---")
    for i, t in enumerate(topics):
        print(f"[{i:02d}] " + ", ".join(t[:topn]))
    print("-" * 60)


def eval_topics(topics: List[List[str]],
                texts: List[List[str]],
                topk: int,
                vocab: List[str],
                processes: int = 1,
                # для эмбеддинговых метрик (опционально)
                w2v_path: Optional[str] = None,
                w2v_binary: bool = True,
                irbo_weight: float = 0.9) -> Dict[str, float]:
    """
    Считает метрики OCTIS для набора тем.
    Всегда: c_v, c_npmi, u_mass, c_uci, TopicDiversity, InvertedRBO.
    Опционально (если задан w2v_path): WECoherence(Pairwise/Centroid),
    WordEmbeddingsInvertedRBO( / Centroid).

    Параметры:
      topics     — список тем, каждая тема — список слов (str)
      texts      — корпус (списки токенов)
      topk       — сколько топ-слов использовать в метриках
      vocab      — словарь корпуса (для фильтрации слов)
      processes  — параллелизм в Coherence (по умолчанию 1)
      w2v_path   — путь к бинарной/текстовой модели эмбеддингов (word2vec/fastText)
      w2v_binary — формат модели (True для .bin)
      irbo_weight — параметр взвешивания RBO (обычно 0.9)
    """
    vocab_set = set(vocab)

    normalized: List[List[str]] = []
    for t in topics:
        if not t:
            continue
        used, clean = set(), []
        for w in t:
            wl = (w or "").strip().lower()
            if wl and wl in vocab_set and wl not in used:
                clean.append(wl);
                used.add(wl)
        if clean:
            normalized.append(clean)

    if not normalized:
        return {
            "coh_c_v": 0.0, "coh_c_npmi": 0.0, "coh_u_mass": 0.0, "coh_c_uci": 0.0,
            "topic_diversity": 0.0, "inv_rbo": 0.0
        }

    k_eff = min(topk, min(len(t) for t in normalized))
    payload = {"topics": [t[:k_eff] for t in normalized]}

    out: Dict[str, float] = {}
    for measure in ("c_v", "c_npmi", "u_mass", "c_uci"):
        out[f"coh_{measure}"] = Coherence(
            texts=texts, topk=k_eff, processes=processes, measure=measure
        ).score(payload)

    out["topic_diversity"] = TopicDiversity(topk=k_eff).score(payload)
    out["inv_rbo"] = InvertedRBO(topk=k_eff, weight=irbo_weight).score(payload)

    if w2v_path:
        out["we_coh_pairwise"] = WECoherencePairwise(
            word2vec_path=w2v_path, binary=w2v_binary, topk=k_eff
        ).score(payload)
        out["we_coh_centroid"] = WECoherenceCentroid(
            word2vec_path=w2v_path, binary=w2v_binary, topk=k_eff
        ).score(payload)
        out["we_inv_rbo"] = WordEmbeddingsInvertedRBO(
            topk=k_eff, weight=irbo_weight, normalize=True,
            word2vec_path=w2v_path, binary=w2v_binary
        ).score(payload)
        out["we_inv_rbo_centroid"] = WordEmbeddingsInvertedRBOCentroid(
            topk=k_eff, weight=irbo_weight, normalize=True,
            word2vec_path=w2v_path, binary=w2v_binary
        ).score(payload)

    return out


def sanitize_for_filename(text: str) -> str:
    """Безопасное имя файла из произвольной строки."""
    if not text:
        return "dataset"
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return out.strip("_") or "dataset"


def write_vw(texts: List[List[str]], path: str) -> None:
    """
    Сохраняет коллекцию в формат Vowpal Wabbit:
      <doc_id> | <term1>:<cnt> <term2>:<cnt> ...
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for i, toks in enumerate(texts):
            if not toks:
                f.write(f"{i} |\n")
                continue
            freq = Counter(toks)
            parts = [f"{w}:{c}" for w, c in freq.items()]
            f.write(f"{i} | {' '.join(parts)}\n")


def load_octis_dataset(name_or_path: str) -> (List[List[str]], List[str]):
    """
    Загрузка корпуса OCTIS:
      1) имя встроенного датасета (например, '20NewsGroup', 'BBC_News', и тд)
      2) или путь к локальной папке в формате OCTIS.
    """
    ds = Dataset()
    if name_or_path and os.path.isdir(name_or_path):
        ds.load_custom_dataset_from_folder(name_or_path)
    else:
        ds.fetch_dataset(name_or_path or "20NewsGroup")
    return ds.get_corpus(), ds.get_vocabulary()
