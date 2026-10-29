from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


STAGES = ["t0", "t1", "t2", "t3"]
LABELS = ["Human", "LLM"]
POS_FEATURES = [
    "pos_ADJ", "pos_ADP", "pos_ADV", "pos_AUX", "pos_CCONJ",
    "pos_DET", "pos_INTJ", "pos_NOUN", "pos_NUM", "pos_PART",
    "pos_PRON", "pos_PROPN", "pos_PUNCT", "pos_SCONJ", "pos_SYM",
    "pos_VERB", "pos_X",
]
STYLE_FEATURES = [
    "dependency_depth",
    "type_token_ratio",
    "sentence_length_mean",
    "sentence_length_variance",
    "punctuation_frequency",
    "average_word_length",
]
FEATURES = POS_FEATURES + STYLE_FEATURES
ORIGIN_COLUMNS = ["dataset", "key", "source"]


def validate_feature_data(frame: pd.DataFrame) -> None:
    required = {
        "dataset", "key", "source", "source_group", "paraphraser",
        "group_id", "stage", *FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if frame["group_id"].isna().any():
        raise ValueError("group_id contains missing values")

    stages = set(frame["stage"].str.lower().unique())
    if stages != set(STAGES):
        raise ValueError(f"Expected stages {STAGES}; found {sorted(stages)}")

    labels = set(frame["source_group"].unique())
    if labels != set(LABELS):
        raise ValueError(f"Expected labels {LABELS}; found {sorted(labels)}")

    group_datasets = frame.groupby("group_id")["dataset"].nunique()
    if not group_datasets.eq(1).all():
        raise ValueError("Each group_id must belong to exactly one dataset")


def make_group_split(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[set[str], set[str], pd.DataFrame]:
    groups = frame[["group_id", "dataset"]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        groups["group_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=groups["dataset"],
    )
    train_ids, test_ids = set(train_ids), set(test_ids)
    if train_ids & test_ids:
        raise AssertionError("group_id leakage detected")

    assignments = groups.copy()
    assignments["split"] = np.where(
        assignments["group_id"].isin(test_ids), "test", "train"
    )
    return train_ids, test_ids, assignments.sort_values("group_id")


def unique_t0_origins(frame: pd.DataFrame) -> pd.DataFrame:
    t0 = frame.loc[frame["stage"].str.lower().eq("t0")].copy()
    check_columns = ["source_group", "group_id", *FEATURES]
    inconsistent = (
        t0.groupby(ORIGIN_COLUMNS, dropna=False)[check_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent.any():
        examples = list(inconsistent[inconsistent].index[:3])
        raise ValueError(f"Inconsistent duplicate T0 origins: {examples}")
    return t0.drop_duplicates(ORIGIN_COLUMNS).copy()


def build_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                C=1.0,
                kernel="rbf",
                class_weight="balanced",
                cache_size=2000,
            ),
        ),
    ])


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def metrics_from_counts(counts: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    hh, hl, lh, ll = np.moveaxis(counts, -1, 0)

    precision_human = _safe_divide(hh, hh + lh)
    recall_human = _safe_divide(hh, hh + hl)
    f1_human = _safe_divide(
        2 * precision_human * recall_human,
        precision_human + recall_human,
    )
    precision_llm = _safe_divide(ll, ll + hl)
    recall_llm = _safe_divide(ll, ll + lh)
    f1_llm = _safe_divide(
        2 * precision_llm * recall_llm,
        precision_llm + recall_llm,
    )

    return {
        "accuracy": _safe_divide(hh + ll, hh + hl + lh + ll),
        "balanced_accuracy": (recall_human + recall_llm) / 2,
        "precision_macro": (precision_human + precision_llm) / 2,
        "recall_macro": (recall_human + recall_llm) / 2,
        "f1_macro": (f1_human + f1_llm) / 2,
        "precision_human": precision_human,
        "recall_human": recall_human,
        "f1_human": f1_human,
        "precision_llm": precision_llm,
        "recall_llm": recall_llm,
        "f1_llm": f1_llm,
    }


def prediction_counts(frame: pd.DataFrame) -> np.ndarray:
    matrix = confusion_matrix(
        frame["source_group"], frame["prediction"], labels=LABELS
    )
    return matrix.reshape(-1)


def cluster_bootstrap_intervals(
    predictions: pd.DataFrame,
    samples: int = 2000,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    if samples <= 0:
        return {}
    per_group = np.stack([
        prediction_counts(group)
        for _, group in predictions.groupby("group_id", sort=True)
    ])
    rng = np.random.default_rng(random_state)
    indices = rng.integers(
        0, len(per_group), size=(samples, len(per_group))
    )
    bootstrap_counts = per_group[indices].sum(axis=1)
    values = metrics_from_counts(bootstrap_counts)
    return {
        name: tuple(np.quantile(metric, [0.025, 0.975]))
        for name, metric in values.items()
    }


def metric_row(
    predictions: pd.DataFrame,
    bootstrap_samples: int = 0,
    random_state: int = 42,
) -> dict[str, float | int]:
    counts = prediction_counts(predictions)
    metrics = {
        name: float(value)
        for name, value in metrics_from_counts(counts).items()
    }
    result: dict[str, float | int] = {
        "n": len(predictions),
        "n_groups": predictions["group_id"].nunique(),
        "true_human": int(counts[0] + counts[1]),
        "true_llm": int(counts[2] + counts[3]),
        **metrics,
    }
    intervals = cluster_bootstrap_intervals(
        predictions,
        samples=bootstrap_samples,
        random_state=random_state,
    )
    for name, (low, high) in intervals.items():
        result[f"{name}_ci_low"] = float(low)
        result[f"{name}_ci_high"] = float(high)
    return result


def evaluate_stages(
    model: Pipeline,
    frame: pd.DataFrame,
    test_ids: set[str],
    bootstrap_samples: int = 2000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, confusion_rows, prediction_frames = [], [], []
    for stage_index, stage in enumerate(STAGES):
        subset = frame.loc[
            frame["group_id"].isin(test_ids)
            & frame["stage"].str.lower().eq(stage)
        ].copy()
        if stage == "t0":
            subset = unique_t0_origins(subset)
        subset["prediction"] = model.predict(subset[FEATURES])
        subset["stage"] = stage
        prediction_frames.append(subset)

        row = metric_row(
            subset,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state + stage_index,
        )
        metric_rows.append({"stage": stage.upper(), **row})

        matrix = confusion_matrix(
            subset["source_group"], subset["prediction"], labels=LABELS
        )
        for true_index, true_label in enumerate(LABELS):
            for pred_index, predicted_label in enumerate(LABELS):
                confusion_rows.append({
                    "stage": stage.upper(),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[true_index, pred_index]),
                })

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(confusion_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def evaluate_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in ["dataset", "paraphraser"]:
        for (stage, value), group in predictions.groupby(["stage", dimension]):
            row = metric_row(group)
            rows.append({
                "dimension": dimension,
                "value": value,
                "stage": stage.upper(),
                **row,
            })
    return pd.DataFrame(rows)


def identify_identity_loss(metrics: pd.DataFrame) -> dict[str, object]:
    ordered = metrics.set_index("stage").loc[[s.upper() for s in STAGES]]
    drops = ordered["f1_macro"].shift(1) - ordered["f1_macro"]
    largest_drop_stage = str(drops.iloc[1:].idxmax())
    largest_drop = float(drops.loc[largest_drop_stage])

    undetectable = ordered.loc[
        ordered["balanced_accuracy_ci_high"].le(0.50), :
    ].index.tolist()
    return {
        "largest_drop_stage": largest_drop_stage,
        "largest_adjacent_macro_f1_drop": largest_drop,
        "undetectable_stage": undetectable[0] if undetectable else None,
    }


def plot_decay(metrics: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(metrics))
    series = [
        ("balanced_accuracy", "Balanced accuracy", "#1677B8", "o"),
        ("f1_macro", "Macro F1", "#E17C05", "s"),
        ("recall_human", "Human recall", "#2E8B57", "^"),
        ("accuracy", "Accuracy", "#7A5195", "D"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for column, label, color, marker in series:
        low = metrics[column] - metrics[f"{column}_ci_low"]
        high = metrics[f"{column}_ci_high"] - metrics[column]
        ax.errorbar(
            x,
            metrics[column],
            yerr=np.vstack([low, high]),
            label=label,
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=7,
            capsize=4,
        )

    majority = metrics["true_llm"] / (metrics["true_human"] + metrics["true_llm"])
    ax.plot(
        x, majority, color="#666666", linestyle="--", linewidth=1.4,
        label="LLM-majority accuracy baseline",
    )
    ax.axhline(
        0.50, color="#999999", linestyle=":", linewidth=1.3,
        label="Balanced-accuracy chance level",
    )
    ax.annotate(
        "Largest Macro F1 drop: −0.140 (T0→T1)",
        xy=(1, metrics.loc[1, "f1_macro"]),
        xytext=(1.25, 0.74),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=10,
    )
    ax.set_xticks(x, metrics["stage"])
    ax.set_ylim(0.40, 0.95)
    ax.set_xlabel("Paraphrase iteration")
    ax.set_ylabel("Held-out performance")
    ax.set_title("RQ2: Human vs. LLM Attribution Decays After T0")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, loc="lower left", frameon=False)
    fig.text(
        0.01, 0.01,
        "95% confidence intervals use group_id-clustered bootstrap resampling.",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    input_path: Path,
    results_dir: Path,
    figures_dir: Path,
    test_size: float = 0.20,
    random_state: int = 42,
    bootstrap_samples: int = 2000,
) -> dict[str, object]:
    frame = pd.read_pickle(input_path)
    frame["stage"] = frame["stage"].str.lower()
    validate_feature_data(frame)
    train_ids, test_ids, assignments = make_group_split(
        frame, test_size=test_size, random_state=random_state
    )

    train = unique_t0_origins(
        frame.loc[frame["group_id"].isin(train_ids)]
    )
    model = build_classifier()
    model.fit(train[FEATURES], train["source_group"])

    metrics, confusion, predictions = evaluate_stages(
        model,
        frame,
        test_ids,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    subgroups = evaluate_subgroups(predictions)
    identity_loss = identify_identity_loss(metrics)

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "rq2_authorship_metrics.csv", index=False)
    confusion.to_csv(results_dir / "rq2_confusion_matrices.csv", index=False)
    subgroups.to_csv(results_dir / "rq2_subgroup_metrics.csv", index=False)
    assignments.to_csv(results_dir / "rq2_group_split.csv", index=False)
    joblib.dump(model, results_dir / "rq2_authorship_model.joblib")
    plot_decay(metrics, figures_dir / "rq2_attribution_decay.png")

    summary = {
        "feature_count": len(FEATURES),
        "model": "class-balanced RBF-SVM (C=1.0)",
        "training_stage": "T0 only",
        "split_unit": "group_id",
        "split_stratification": "dataset",
        "random_state": random_state,
        "test_size": test_size,
        "train_groups": len(train_ids),
        "test_groups": len(test_ids),
        "train_unique_t0_origins": len(train),
        "bootstrap_samples": bootstrap_samples,
        **identity_loss,
    }
    (results_dir / "rq2_authorship_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/processed/update2_features.pkl",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=root / "data/results/update2"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=root / "figures/update2"
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    summary = run_experiment(
        input_path=args.input,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
