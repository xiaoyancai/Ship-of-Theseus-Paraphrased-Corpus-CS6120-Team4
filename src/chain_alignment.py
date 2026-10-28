
from pathlib import Path
import pandas as pd

from src.preprocessing import load_all_datasets, normalize_versions

# Paraphraser names and their version tokens
PARAPHRASERS = {
    "ChatGPT": "chatgpt",
    "PaLM": "palm",
    "Dipper": "dipper",
    "Dipper(low)": "dipper(low)",
    "Dipper(high)": "dipper(high)",
    "Pegasus(slight)": "pegasus(slight)",
    "Pegasus(full)": "pegasus(full)",
}

ID_COLS = ["dataset", "key", "source"]
STAGES = ["t0", "t1", "t2", "t3"]


def build_update2_chains(df):
    """Build aligned T0–T3 chains for all sources and paraphrasers."""
    df = normalize_versions(df)

    all_chains = []

    for name, token in PARAPHRASERS.items():
        # Map each version string to its iteration
        version_map = {"original": "t0"}
        for n in range(1, 4):
            version_map["_".join([token] * n)] = f"t{n}"

        subset = df[df["version_name"].isin(version_map)].copy()
        subset["stage"] = subset["version_name"].map(version_map)

        # Ensure each source document has at most one text per stage
        if subset.duplicated(ID_COLS + ["stage"]).any():
            raise ValueError(f"Duplicate document versions: {name}")

        # Align T0, T1, T2, and T3
        chains = (
            subset.pivot(
                index=ID_COLS,
                columns="stage",
                values="text"
            )
            .reindex(columns=STAGES)
            .reset_index()
        )

        # Keep complete, nonempty chains
        chains = chains.dropna(subset=STAGES).copy()

        valid = chains[STAGES].apply(
            lambda col: col.astype(str).str.strip().ne("")
        ).all(axis=1)

        chains = chains.loc[valid].copy()
        chains["paraphraser"] = name

        all_chains.append(
            chains[ID_COLS + ["paraphraser"] + STAGES]
        )

        print(f"{name}: {len(chains):,} complete chains")

    result = pd.concat(all_chains, ignore_index=True)

    # Labels used by the later experiments
    result["source_group"] = result["source"].apply(
        lambda x: "Human" if x == "Human" else "LLM"
    )

    # Keep versions of the same document in one train/test split
    result["group_id"] = (
        result["dataset"].astype(str)
        + "::"
        + result["key"].astype(str)
    )

    # Verify uniqueness and original-text consistency
    assert not result.duplicated(
        ID_COLS + ["paraphraser"]
    ).any()

    assert (
        result.groupby(ID_COLS)["t0"].nunique() == 1
    ).all()

    return result


def main():
    # Reuse the existing seven-dataset loader
    raw = load_all_datasets()

    chains = build_update2_chains(raw)

    # Save the shared Update 2 dataset
    output = Path("data/processed/update2_chains.pkl")
    output.parent.mkdir(parents=True, exist_ok=True)
    chains.to_pickle(output)

    # Summarize dataset availability
    summary = (
        chains.groupby(
            ["dataset", "source_group", "paraphraser"]
        )
        .size()
        .reset_index(name="count")
    )

    summary_path = Path("data/metadata/update2_chain_counts.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print("\nFinal shape:", chains.shape)
    print("Datasets:", chains["dataset"].nunique())
    print("Sources:", chains["source"].nunique())
    print("Paraphrasers:", chains["paraphraser"].nunique())
    print("\nSource groups:")
    print(chains["source_group"].value_counts())
    print("\nSample:")
    print(chains.head(2))


if __name__ == "__main__":
    main()
