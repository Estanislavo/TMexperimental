import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
import time
import tempfile
import uuid
import shutil
from typing import List, Tuple, Optional, Dict

import pandas as pd

from .utils import (
    pretty_print_topics,
    eval_topics,
    sanitize_for_filename,
    write_vw,
    load_octis_dataset,
)

HAVE_GENSIM = True
try:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel, LdaMulticore
except Exception:
    HAVE_GENSIM = False

HAVE_ARTM = True
ARTM_IMPORT_ERROR = None
try:
    try:
        import artm  # type: ignore
    except Exception as e:
        if "Descriptors cannot be created directly" in repr(e):
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            import importlib as _il

            _il.invalidate_caches()
            import artm  # retry
        else:
            raise
except Exception as e:
    HAVE_ARTM = False
    ARTM_IMPORT_ERROR = e

HAVE_TOP2VEC = True
try:
    from top2vec import Top2Vec
except Exception:
    HAVE_TOP2VEC = False


def train_gensim_lda(
        texts: List[List[str]],
        K: int,
        passes: int,
        iterations: int,
        chunksize: int,
        workers: int
) -> Tuple[Optional[object], float, List[List[str]]]:
    """Обучает LDA (gensim) и возвращает (модель, время_сек, темы)."""
    if not HAVE_GENSIM:
        return None, 0.0, []

    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100_000)
    corpus = [dictionary.doc2bow(t) for t in texts]

    start = time.perf_counter()
    if workers and workers > 1:
        lda = LdaMulticore(
            corpus=corpus, id2word=dictionary, num_topics=K,
            passes=passes, iterations=iterations, chunksize=chunksize,
            random_state=42, eval_every=None,
            alpha="symmetric", eta=None, workers=workers
        )
    else:
        lda = LdaModel(
            corpus=corpus, id2word=dictionary, num_topics=K,
            passes=passes, iterations=iterations, chunksize=chunksize,
            random_state=42, eval_every=None,
            alpha="auto", eta="auto"
        )
    dur = time.perf_counter() - start

    shown = lda.show_topics(num_topics=K, num_words=50, formatted=False)
    topics = [[w for (w, _) in words] for _, words in shown]
    return lda, dur, topics


def train_bigartm(
        texts: List[List[str]],
        K: int,
        passes: int,
        procs: int,
        tmpdir: Optional[str] = None
) -> Tuple[Optional[object], float, List[List[str]]]:
    """
    Обучает BigARTM в оффлайн-режиме:
      1) пишет VW,
      2) строит батчи,
      3) обучает,
      4) вытаскивает топ-слова из φ.
    """
    if not HAVE_ARTM:
        return None, 0.0, []

    base_tmp = tmpdir or tempfile.gettempdir()
    vw_path = os.path.join(base_tmp, "vw_octis", "corpus.vw")
    write_vw(texts, vw_path)

    batches_dir = os.path.join(base_tmp, f"artm_batches_{uuid.uuid4().hex}")
    os.makedirs(batches_dir, exist_ok=True)

    start = time.perf_counter()
    bv = artm.BatchVectorizer(
        data_path=vw_path,
        data_format="vowpal_wabbit",
        target_folder=batches_dir
    )
    model = artm.ARTM(num_topics=K, num_processors=procs, dictionary=bv.dictionary)
    model.fit_offline(batch_vectorizer=bv, num_collection_passes=passes)
    dur = time.perf_counter() - start

    phi = model.get_phi()
    topics = [phi[col].sort_values(ascending=False).head(50).index.tolist() for col in phi.columns]

    try:
        shutil.rmtree(batches_dir)
    except Exception:
        pass

    return model, dur, topics


def train_ct2v(
        texts: List[List[str]],
        K: int,
        embed_model: str,
        workers: int,
        batch_size: int
) -> Tuple[Optional[object], float, List[List[str]]]:
    if not HAVE_TOP2VEC:
        return None, 0.0, []

    docs = [" ".join(t) for t in texts]
    start = time.perf_counter()
    model = Top2Vec(
        documents=docs,
        contextual_top2vec=True,
        embedding_model=embed_model,
        speed="learn",
        workers=workers,
        embedding_batch_size=batch_size,
        min_count=1
    )

    try:
        n = model.get_num_topics()
        if n > K:
            if hasattr(model, "hierarchical_topic_reduction"):
                model.hierarchical_topic_reduction(num_topics=K)
            else:
                model.reduce_topics(num_topics=K)
    except Exception:
        pass

    dur = time.perf_counter() - start

    try:
        res = model.get_topics()
        tw = res[0] if isinstance(res, tuple) else res
        topics = [[(w or "").lower() for w in words][:50] for words in tw][:K]
    except Exception:
        topics = []

    return model, dur, topics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="20NewsGroup",
                    help="имя OCTIS датасета (20NewsGroup, BBC_News, ...) или путь к локальной папке OCTIS")
    ap.add_argument("--topics", type=int, default=20)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--print-topn", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=0)

    # LDA
    ap.add_argument("--lda-passes", type=int, default=10)
    ap.add_argument("--lda-iters", type=int, default=50)
    ap.add_argument("--lda-chunksize", type=int, default=2000)

    # BigARTM
    ap.add_argument("--artm-passes", type=int, default=10)
    ap.add_argument("--artm-procs", type=int, default=4)
    ap.add_argument("--tmpdir", type=str, default=None)

    # CT2V
    ap.add_argument("--no-ct2v", action="store_true", help="не запускать ContextualTop2Vec")
    ap.add_argument("--ct2v-embed", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--ct2v-batch", type=int, default=8)

    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    try:
        texts, vocab = load_octis_dataset(args.dataset)
    except Exception as e:
        print(f"[ERROR] не удалось загрузить датасет '{args.dataset}': {e}")
        sys.exit(2)

    if args.subset and args.subset > 0:
        texts = texts[:args.subset]
        print(f"[INFO] subset: используем первые {len(texts)} документов")

    results: List[Dict[str, object]] = []
    dataset_tag = sanitize_for_filename(args.dataset)

    # LDA
    if HAVE_GENSIM:
        print("=== gensim LDA ===")
        _, sec, tops = train_gensim_lda(
            texts, args.topics, args.lda_passes, args.lda_iters, args.lda_chunksize, args.workers
        )
        pretty_print_topics("gensim_LDA", tops, args.print_topn)
        met = eval_topics(tops, texts, args.topk, vocab)
        row = {"dataset": args.dataset, "model": "gensim_LDA", "K": args.topics, "time_sec": round(sec, 3)}
        row.update({k: round(v, 6) for k, v in met.items()})
        results.append(row);
        print(row)
    else:
        print("[WARN] gensim не установлен — пропускаю LDA")

    # BigARTM
    if HAVE_ARTM:
        print("=== BigARTM ===")
        _, sec, tops = train_bigartm(
            texts, args.topics, args.artm_passes, args.artm_procs, args.tmpdir
        )
        pretty_print_topics("BigARTM", tops, args.print_topn)
        met = eval_topics(tops, texts, args.topk, vocab)
        row = {"dataset": args.dataset, "model": "BigARTM", "K": args.topics, "time_sec": round(sec, 3)}
        row.update({k: round(v, 6) for k, v in met.items()})
        results.append(row);
        print(row)
    else:
        print("[WARN] BigARTM не установлен — пропускаю BigARTM")
        if ARTM_IMPORT_ERROR:
            print("[WARN] причина импорта BigARTM:", repr(ARTM_IMPORT_ERROR))

    # ContextualTop2Vec
    if not args.no_ct2v and HAVE_TOP2VEC:
        print("=== ContextualTop2Vec ===")
        _, sec, tops = train_ct2v(
            texts, args.topics, args.ct2v_embed, args.workers, args.ct2v_batch
        )
        pretty_print_topics("ContextualTop2Vec", tops, args.print_topn)
        met = eval_topics(tops, texts, args.topk, vocab)
        row = {"dataset": args.dataset, "model": "ContextualTop2Vec", "K": len(tops), "time_sec": round(sec, 3)}
        row.update({k: round(v, 6) for k, v in met.items()})
        results.append(row);
        print(row)
    elif not args.no_ct2v and not HAVE_TOP2VEC:
        print("[WARN] top2vec не установлен — пропускаю ContextualTop2Vec")

    if not results:
        print("Ни одна модель не запущена. Проверь зависимости и флаги.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, f"topic_benchmark_{dataset_tag}_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
