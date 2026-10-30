from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


STAGES = ["t0", "t1", "t2", "t3"]
IDENTITY_COLUMNS = ["dataset", "group_id", "source_group", "stage"]
POS_FEATURES = [
    "pos_ADJ", "pos_ADP", "pos_ADV", "pos_AUX", "pos_CCONJ",
    "pos_DET", "pos_INTJ", "pos_NOUN", "pos_NUM", "pos_PART",
    "pos_PRON", "pos_PROPN", "pos_PUNCT", "pos_SCONJ", "pos_SYM",
    "pos_VERB", "pos_X",
]
STYLE_FEATURES = [
    "dependency_depth", "type_token_ratio", "sentence_length_mean",
    "sentence_length_variance", "punctuation_frequency",
    "average_word_length",
]
FEATURES = POS_FEATURES + STYLE_FEATURES


def validate_feature_frame(frame: pd.DataFrame) -> None:
    required = set(IDENTITY_COLUMNS + FEATURES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["group_id"].isna().any():
        raise ValueError("group_id contains missing values")
    if not frame["source_group"].isin(["Human", "LLM"]).all():
        raise ValueError("source_group must contain only Human and LLM")
    if set(frame["stage"].str.lower().unique()) != set(STAGES):
        raise ValueError("Expected stages t0, t1, t2, and t3")
    group_datasets = frame.groupby("group_id")["dataset"].nunique()
    if not group_datasets.eq(1).all():
        raise ValueError("Each group_id must belong to exactly one dataset")
    if np.isinf(frame[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Feature matrix contains infinite values")


def make_group_split(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[set[str], set[str], pd.DataFrame]:
    groups = frame[["group_id", "dataset"]].drop_duplicates()
    if groups["group_id"].duplicated().any():
        raise ValueError("group_id maps to multiple datasets")
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
    return train_ids, test_ids, assignments.sort_values("group_id").reset_index(drop=True)


def aggregate_group_stage(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] = FEATURES,
) -> pd.DataFrame:
    keys = ["dataset", "group_id", "source_group", "stage"]
    result = (
        frame.assign(stage=frame["stage"].str.lower())
        .groupby(keys, as_index=False, sort=False)[list(feature_columns)]
        .mean()
    )
    result["stage"] = pd.Categorical(result["stage"], categories=STAGES, ordered=True)
    return result.sort_values(keys).reset_index(drop=True)


def fit_pca_projection(
    aggregated: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] = FEATURES,
    train_group_ids: set[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, SimpleImputer, StandardScaler, PCA]:
    if train_group_ids is None:
        train_group_ids = set(aggregated["group_id"])
    train_mask = aggregated["group_id"].isin(train_group_ids)
    imputer = SimpleImputer(strategy="median").fit(aggregated.loc[train_mask, feature_columns])
    imputed_train = imputer.transform(aggregated.loc[train_mask, feature_columns])
    imputed = imputer.transform(aggregated[feature_columns])
    scaler = StandardScaler().fit(imputed_train)
    scaled = scaler.transform(imputed)
    pca = PCA(n_components=2, random_state=random_state).fit(scaled[train_mask])
    result = aggregated[IDENTITY_COLUMNS].copy()
    result["split"] = np.where(train_mask, "train", "test")
    result["pca1"], result["pca2"] = pca.transform(scaled).T
    return result, imputer, scaler, pca


def sample_balanced(
    frame: pd.DataFrame,
    train_group_ids: set[str],
    per_cell: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    train = frame[frame["group_id"].isin(train_group_ids)].copy()
    parts = []
    for cell_index, (_, group) in enumerate(
        train.groupby(["dataset", "source_group", "stage"], observed=True)
    ):
        parts.append(group.sample(min(per_cell, len(group)), random_state=random_state + cell_index))
    if not parts:
        raise ValueError("No rows available for t-SNE sampling")
    return pd.concat(parts, ignore_index=True)


def fit_tsne_projection(
    sampled: pd.DataFrame,
    aggregated: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: list[str] | tuple[str, ...] = FEATURES,
    random_state: int = 42,
    perplexity: float = 30.0,
) -> pd.DataFrame:
    values = aggregated.merge(
        sampled[IDENTITY_COLUMNS], on=IDENTITY_COLUMNS, how="inner", validate="one_to_one"
    )
    feature_values = values[list(feature_columns)]
    scaled = scaler.transform(imputer.transform(feature_values))
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
        random_state=random_state,
    )
    coordinates = model.fit_transform(scaled)
    result = values[IDENTITY_COLUMNS].copy()
    result["tsne1"], result["tsne2"] = coordinates.T
    return result


def summarize_separation(
    embedded: pd.DataFrame,
    coordinate_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for stage in STAGES:
        stage_frame = embedded[embedded["stage"].astype(str).eq(stage)]
        dataset_centroids = (
            stage_frame.groupby(["dataset", "source_group"], as_index=False)[list(coordinate_columns)]
            .mean()
        )
        source_centroids = dataset_centroids.groupby("source_group")[list(coordinate_columns)].mean()
        human = source_centroids.loc["Human"].to_numpy()
        llm = source_centroids.loc["LLM"].to_numpy()
        row = {
            "stage": stage,
            "n_datasets": dataset_centroids["dataset"].nunique(),
            "centroid_distance": float(np.linalg.norm(human - llm)),
        }
        for label, values in [("human", human), ("llm", llm)]:
            for index, coordinate in enumerate(coordinate_columns, start=1):
                row[f"{label}_{coordinate}"] = float(values[index - 1])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pca_trajectory(embedded: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"Human": "#2f6690", "LLM": "#d95f02"}
    for source_group, group in embedded.groupby("source_group"):
        centroid = group.groupby("stage", observed=False)[["pca1", "pca2"]].mean().reindex(STAGES)
        ax.plot(centroid.pca1, centroid.pca2, marker="o", linewidth=2, label=source_group, color=colors[source_group])
        for stage, row in centroid.iterrows():
            ax.annotate(stage.upper(), (row.pca1, row.pca2), xytext=(4, 4), textcoords="offset points")
    ax.set(title="Identity trajectory in PCA space", xlabel="PC1", ylabel="PC2")
    ax.legend(title="Original source")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tsne_trajectory(embedded: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = dict(zip(STAGES, ["o", "s", "^", "D"]))
    colors = {"Human": "#2f6690", "LLM": "#d95f02"}
    for (source_group, stage), group in embedded.groupby(["source_group", "stage"], observed=True):
        ax.scatter(group.tsne1, group.tsne2, s=18, alpha=0.55, marker=markers[str(stage)],
                   color=colors[source_group], label=f"{source_group} {str(stage).upper()}")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.set(title="Identity trajectory in t-SNE space", xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(unique.values(), unique.keys(), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_analysis(
    features_path: Path,
    results_dir: Path,
    figures_dir: Path,
    random_state: int = 42,
    tsne_per_cell: int = 50,
) -> dict[str, object]:
    features = pd.read_pickle(features_path)
    validate_feature_frame(features)
    train_ids, test_ids, split = make_group_split(features, random_state=random_state)
    aggregated = aggregate_group_stage(features)
    pca_frame, imputer, scaler, pca = fit_pca_projection(
        aggregated, train_group_ids=train_ids, random_state=random_state
    )
    sampled = sample_balanced(aggregated, train_ids, per_cell=tsne_per_cell, random_state=random_state)
    tsne_frame = fit_tsne_projection(
        sampled, aggregated, imputer, scaler, random_state=random_state
    )
    separation = summarize_separation(pca_frame, ["pca1", "pca2"])
    separation.to_csv(results_dir / "rq2_identity_separation.csv", index=False)
    pca_frame.to_csv(results_dir / "rq2_pca_embeddings.csv", index=False)
    tsne_frame.to_csv(results_dir / "rq2_tsne_embeddings.csv", index=False)
    split.to_csv(results_dir / "rq2_identity_group_split.csv", index=False)
    plot_pca_trajectory(pca_frame, figures_dir / "rq2_pca_identity_trajectory.png")
    plot_tsne_trajectory(tsne_frame, figures_dir / "rq2_tsne_identity_trajectory.png")
    summary = {
        "n_feature_rows": int(len(features)),
        "n_aggregated_rows": int(len(aggregated)),
        "n_groups": int(aggregated.group_id.nunique()),
        "n_train_groups": int(len(train_ids)),
        "n_test_groups": int(len(test_ids)),
        "feature_count": len(FEATURES),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "tsne_sample_rows": int(len(tsne_frame)),
        "random_state": random_state,
    }
    (results_dir / "rq2_identity_trajectory_summary.json").write_text(json.dumps(summary, indent=2))
    return {"pca": pca_frame, "tsne": tsne_frame, "separation": separation, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    results_dir = args.project_root / "data/results/update2"
    figures_dir = args.project_root / "figures/update2"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    run_analysis(args.project_root / "data/processed/update2_features.pkl", results_dir, figures_dir)


if __name__ == "__main__":
    main()
