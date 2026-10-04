import pandas as pd

from src.similarity_baseline import METRIC_COLUMNS, score_dataset, score_pair, summarize_scores


def test_score_pair_returns_expected_metrics_in_range():
    scores = score_pair(
        "The cat sat on the mat.",
        "The cat sat on a mat.",
        bertscore_fn=lambda refs, cands, **_: ([0.9], [0.8], [0.85]),
    )

    assert set(scores) == set(METRIC_COLUMNS)
    assert all(0 <= scores[name] <= 1 for name in METRIC_COLUMNS)


def test_score_dataset_scores_all_generations_and_preserves_metadata():
    frame = pd.DataFrame([{
        "dataset": "Yelp", "key": "y-1", "source": "Human", "paraphraser": "ChatGPT",
        "t0": "A short review.", "t1": "A brief review.", "t2": "A concise review.", "t3": "A small review.",
    }])
    result = score_dataset(
        frame,
        bertscore_fn=lambda refs, cands, **_: ([0.9] * len(refs), [0.8] * len(refs), [0.85] * len(refs)),
    )

    assert result[["dataset", "key", "paraphraser", "generation"]].to_dict("records") == [
        {"dataset": "Yelp", "key": "y-1", "paraphraser": "ChatGPT", "generation": "t1"},
        {"dataset": "Yelp", "key": "y-1", "paraphraser": "ChatGPT", "generation": "t2"},
        {"dataset": "Yelp", "key": "y-1", "paraphraser": "ChatGPT", "generation": "t3"},
    ]
    assert result[list(METRIC_COLUMNS)].notna().all().all()


def test_summarize_scores_has_mean_std_and_count():
    details = pd.DataFrame([
        {"dataset": "Yelp", "paraphraser": "ChatGPT", "generation": "t1", "bleu": 0.5, "rouge1_f1": 0.6, "rouge2_f1": 0.4, "rougeL_f1": 0.55, "bertscore_precision": 0.8, "bertscore_recall": 0.7, "bertscore_f1": 0.75},
        {"dataset": "Yelp", "paraphraser": "ChatGPT", "generation": "t1", "bleu": 0.7, "rouge1_f1": 0.8, "rouge2_f1": 0.6, "rougeL_f1": 0.75, "bertscore_precision": 0.9, "bertscore_recall": 0.8, "bertscore_f1": 0.85},
    ])
    summary = summarize_scores(details)

    assert len(summary) == 1
    assert summary.loc[0, "count"] == 2
    assert summary.loc[0, "bleu_mean"] == 0.6
    assert summary.loc[0, "bleu_std"] > 0


def test_lexical_only_mode_skips_bertscore():
    frame = pd.DataFrame([{
        "dataset": "XSum", "key": "x-1", "source": "Human", "paraphraser": "Dipper",
        "t0": "A source.", "t1": "A rewrite.", "t2": "Another rewrite.", "t3": "A final rewrite.",
    }])
    result = score_dataset(frame, metrics=("lexical",), bertscore_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("BERTScore called")))

    assert set(result.columns) == {"dataset", "key", "source", "paraphraser", "generation", "bleu", "rouge1_f1", "rouge2_f1", "rougeL_f1"}


def test_bertscore_only_mode_skips_lexical():
    frame = pd.DataFrame([{
        "dataset": "XSum", "key": "x-2", "source": "Human", "paraphraser": "Dipper",
        "t0": "A source.", "t1": "A rewrite.", "t2": "Another rewrite.", "t3": "A final rewrite.",
    }])
    result = score_dataset(
        frame,
        metrics=("bertscore",),
        bertscore_fn=lambda refs, cands, **_: ([0.9] * len(refs), [0.8] * len(refs), [0.85] * len(refs)),
    )

    assert set(result.columns) == {"dataset", "key", "source", "paraphraser", "generation", "bertscore_precision", "bertscore_recall", "bertscore_f1"}
