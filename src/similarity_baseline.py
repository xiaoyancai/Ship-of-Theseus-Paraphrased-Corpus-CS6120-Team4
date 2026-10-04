"""Compute T0-to-T1/T2/T3 BLEU, ROUGE, and BERTScore baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


REQUIRED_COLUMNS = {"dataset", "key", "source", "paraphraser", "t0", "t1", "t2", "t3"}
GENERATIONS = ("t1", "t2", "t3")
METRIC_COLUMNS = ("bleu", "rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_precision", "bertscore_recall", "bertscore_f1")
LEXICAL_COLUMNS = ("bleu", "rouge1_f1", "rouge2_f1", "rougeL_f1")
BERTSCORE_COLUMNS = ("bertscore_precision", "bertscore_recall", "bertscore_f1")
ROUGE_SCORER = None
BERT_SCORER = None


def _lexical_scores(reference: str, candidate: str) -> dict[str, float]:
    global ROUGE_SCORER
    import sacrebleu
    from rouge_score import rouge_scorer

    if ROUGE_SCORER is None:
        ROUGE_SCORER = rouge_scorer.RougeScorer(("rouge1", "rouge2", "rougeL"), use_stemmer=True)
    rouge_scores = ROUGE_SCORER.score(reference, candidate)
    return {
        "bleu": float(sacrebleu.sentence_bleu(candidate, [reference]).score / 100.0),
        "rouge1_f1": float(rouge_scores["rouge1"].fmeasure),
        "rouge2_f1": float(rouge_scores["rouge2"].fmeasure),
        "rougeL_f1": float(rouge_scores["rougeL"].fmeasure),
    }


def _default_bertscore(references, candidates, *, model_type, lang, batch_size, device):
    global BERT_SCORER
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError("BERTScore is unavailable. Install dependencies with `pip install -r requirements.txt`.") from exc
    if BERT_SCORER is None:
        BERT_SCORER = BERTScorer(model_type=model_type, lang=lang, rescale_with_baseline=False, device=device)
    precision, recall, f1 = BERT_SCORER.score(candidates, references, batch_size=batch_size, verbose=False)
    return precision.detach().cpu().tolist(), recall.detach().cpu().tolist(), f1.detach().cpu().tolist()


def score_pair(reference, candidate, *, bertscore_fn=None, model_type="distilbert-base-uncased", lang="en", batch_size=8, device="cpu"):
    """Score one candidate against the original T0 reference."""
    scores = _lexical_scores(reference, candidate)
    bertscore_fn = bertscore_fn or _default_bertscore
    precision, recall, f1 = bertscore_fn([reference], [candidate], model_type=model_type, lang=lang, batch_size=batch_size, device=device)
    scores.update({"bertscore_precision": float(precision[0]), "bertscore_recall": float(recall[0]), "bertscore_f1": float(f1[0])})
    return scores


def _validate_input(frame):
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Input contains missing values in required columns")


def _score_bertscore_in_chunks(references, candidates, *, bertscore_fn, model_type, lang, batch_size, device):
    """Score bounded batches so CPU memory does not grow with corpus size."""
    chunk_size = max(batch_size, batch_size * 16)
    precision, recall, f1 = [], [], []
    for start in range(0, len(references), chunk_size):
        chunk_precision, chunk_recall, chunk_f1 = bertscore_fn(
            references[start:start + chunk_size], candidates[start:start + chunk_size],
            model_type=model_type, lang=lang, batch_size=batch_size, device=device,
        )
        precision.extend(chunk_precision)
        recall.extend(chunk_recall)
        f1.extend(chunk_f1)
    return precision, recall, f1


def score_dataset(frame, *, metrics=("lexical", "bertscore"), bertscore_fn=None, model_type="distilbert-base-uncased", lang="en", batch_size=8, device="cpu"):
    """Return one metric row for every document and generation T1/T2/T3."""
    _validate_input(frame)
    metrics = tuple(metrics)
    if not set(metrics).issubset({"lexical", "bertscore"}) or not metrics:
        raise ValueError("metrics must contain lexical and/or bertscore")
    bertscore_fn = bertscore_fn or _default_bertscore
    rows = []
    references = frame["t0"].astype(str).tolist()
    lexical_by_generation = {}
    if "bertscore" in metrics:
        all_references = []
        all_candidates = []
        for generation in GENERATIONS:
            candidates = frame[generation].astype(str).tolist()
            all_references.extend(references)
            all_candidates.extend(candidates)
        precision, recall, f1 = _score_bertscore_in_chunks(
            all_references, all_candidates, bertscore_fn=bertscore_fn,
            model_type=model_type, lang=lang, batch_size=batch_size, device=device,
        )
    if "lexical" in metrics:
        lexical_by_generation = {
            generation: [_lexical_scores(reference, candidate) for reference, candidate in zip(references, frame[generation].astype(str).tolist())]
            for generation in GENERATIONS
        }
    for generation_index, generation in enumerate(GENERATIONS):
        lexical = lexical_by_generation.get(generation, {})
        offset = generation_index * len(frame)
        for index, (_, source_row) in enumerate(frame.iterrows()):
            row = {
                "dataset": source_row["dataset"], "key": source_row["key"], "source": source_row["source"],
                "paraphraser": source_row["paraphraser"], "generation": generation,
            }
            if "lexical" in metrics:
                row.update(lexical[index])
            if "bertscore" in metrics:
                row.update({
                    "bertscore_precision": float(precision[offset + index]),
                    "bertscore_recall": float(recall[offset + index]),
                    "bertscore_f1": float(f1[offset + index]),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_scores(details):
    """Aggregate metric mean, sample standard deviation, and count by group."""
    group_columns = ["dataset", "paraphraser", "generation"]
    available_metrics = [column for column in METRIC_COLUMNS if column in details.columns]
    summary = details.groupby(group_columns, dropna=False)[available_metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(column).rstrip("_") if isinstance(column, tuple) else column for column in summary.columns]
    counts = details.groupby(group_columns, dropna=False).size().reset_index(name="count")
    return summary.merge(counts, on=group_columns, how="left")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/update1_processed.csv"))
    parser.add_argument("--details-output", type=Path, default=Path("data/results/update1_similarity_scores.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/results/update1_similarity_summary.csv"))
    parser.add_argument("--model-type", default="distilbert-base-uncased")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="BERTScore device, e.g. cpu or cuda")
    parser.add_argument("--metrics", nargs="+", choices=("lexical", "bertscore"), default=("lexical", "bertscore"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows for a pilot run")
    return parser


def main(argv: Iterable[str] | None = None):
    args = _build_parser().parse_args(argv)
    frame = pd.read_csv(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        frame = frame.head(args.limit)
    details = score_dataset(frame, metrics=args.metrics, model_type=args.model_type, lang=args.lang, batch_size=args.batch_size, device=args.device)
    summary = summarize_scores(details)
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(f"Scored {len(details):,} rows across {len(summary):,} groups.")
    print(f"Detailed results: {args.details_output}")
    print(f"Summary results: {args.summary_output}")


if __name__ == "__main__":
    main()
