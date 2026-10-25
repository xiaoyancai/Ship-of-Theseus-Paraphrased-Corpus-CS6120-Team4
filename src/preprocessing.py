import pandas as pd
from pathlib import Path

# Version Mappings
CHATGPT_VERSIONS = {
    "original": "t0",
    "chatgpt": "t1",
    "chatgpt_chatgpt": "t2",
    "chatgpt_chatgpt_chatgpt": "t3",
}

PALM_VERSIONS = {
    "original": "t0",
    "palm": "t1",
    "palm_palm": "t2",
    "palm_palm_palm": "t3",
}

PEGASUS_VERSIONS = {
    "original": "t0",
    "pegasus(full)": "t1",
    "pegasus(full)_pegasus(full)": "t2",
    "pegasus(full)_pegasus(full)_pegasus(full)": "t3",
}

DIPPER_VERSIONS = {
    "original": "t0",
    "dipper": "t1",
    "dipper_dipper": "t2",
    "dipper_dipper_dipper": "t3",
}

# T0 and T1 versions for the Linguistic Forensic Audit
AUDIT_VERSIONS = {
    "original": "text_T0",
    "chatgpt": "text_chatgpt",
    "palm": "text_palm",
    "dipper": "text_dipper",
    "dipper(low)": "text_dipper_low",
    "dipper(high)": "text_dipper_high",
    "pegasus(slight)": "text_pegasus_slight",
    "pegasus(full)": "text_pegasus_full",
}


'''
build_complete_chain() is the function 
to takes the long-format raw paraphrased data for one paraphraser 
and turns it into a clean one-row-per-document T0→T3 table.

Parameters
----------
df : pandas.DataFrame
    Input dataframe containing at least:
    source, key, text, version_name.

dataset_name : str
    Dataset name, e.g. "yelp" or "xsum".

paraphraser_name : str
    Display name, e.g. "ChatGPT".

version_map : dict
    Mapping from version_name to t0/t1/t2/t3.

Returns
-------
pandas.DataFrame
    Complete chains with columns:
    dataset, key, source, paraphraser, t0, t1, t2, t3.
'''


def build_complete_chain(
    df,
    dataset_name,
    paraphraser_name,
    version_map
):
    # Keep only versions used in this paraphrasing chain
    temp = df[
        df["version_name"].isin(version_map.keys())
    ].copy()

    # Map raw version names to t0/t1/t2/t3
    temp["stage"] = temp["version_name"].map(version_map)

    duplicates = (
        temp
        .groupby(["key", "stage"])
        .size()
        .reset_index(name="count")
    )

    if (duplicates["count"] > 1).any():
        print(
            f"Warning: duplicate key-stage rows found for "
            f"{dataset_name} / {paraphraser_name}"
        )

    # Reshape to one row per document: key | t0 | t1 | t2 | t3
    chain = (
        temp
        .pivot(
            index="key",
            columns="stage",
            values="text"
        )
        .reset_index()
    )

    # Remove the pivot column-axis label
    chain.columns.name = None

    # Keep only documents with all four stages
    complete = chain.dropna(
        subset=["t0", "t1", "t2", "t3"]
    ).copy()

    # Add dataset metadata
    complete["dataset"] = dataset_name
    complete["source"] = "Human"
    complete["paraphraser"] = paraphraser_name

    # Keep a consistent output column order
    complete = complete[
        [
            "dataset",
            "key",
            "source",
            "paraphraser",
            "t0",
            "t1",
            "t2",
            "t3",
        ]
    ]

    # Print candidate and complete-chain counts
    print(
        f"{dataset_name} / {paraphraser_name}"
        f" | candidates: {len(chain)}"
        f" | complete: {len(complete)}"
    )

    return complete




def process_dataset(df, dataset_name):
    """
    Process one dataset for the four main paraphrasers.
    """

    human_df = df[
        df["source"] == "Human"
    ].copy()

    chatgpt = build_complete_chain(
        human_df,
        dataset_name,
        "ChatGPT",
        CHATGPT_VERSIONS,
    )

    palm = build_complete_chain(
        human_df,
        dataset_name,
        "PaLM",
        PALM_VERSIONS,
    )

    pegasus = build_complete_chain(
        human_df,
        dataset_name,
        "Pegasus",
        PEGASUS_VERSIONS,
    )

    dipper = build_complete_chain(
        human_df,
        dataset_name,
        "Dipper",
        DIPPER_VERSIONS,
    )

    processed = pd.concat(
        [chatgpt, palm, pegasus, dipper],
        ignore_index=True,
    )

    return processed


# All datasets required for the Linguistic Forensic Audit
DATASETS = [
    "xsum",
    "tldr",
    "sci_gen",
    "cmv",
    "wp",
    "eli5",
    "yelp",
]


def load_all_datasets(data_dir="data/raw/paraphrased_datasets"):
    """
    Load and concatenate all seven datasets.
    Keep both Human and LLM sources.
    """
    frames = []

    for name in DATASETS:
        file_path = Path(data_dir) / f"{name}_paraphrased.csv"

        df = pd.read_csv(file_path)
        df["dataset"] = name

        frames.append(df)
        print(f"{name}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)

    print(f"\nTotal rows: {len(combined):,}")
    print(f"Sources: {combined['source'].unique()}")

    return combined


def normalize_versions(df):
    """Normalize original version labels."""
    df = df.copy()

    df["version_name"] = df["version_name"].replace({
        "orignal": "original"
    })

    return df



def build_paired_dataset(df):
    """
    Align original and T1 paraphrased texts for each article-source pair.
    """
    df = normalize_versions(df)

    # Select T0 and all seven T1 paraphraser variants
    temp = df[df["version_name"].isin(AUDIT_VERSIONS)].copy()
    temp["text_column"] = temp["version_name"].map(AUDIT_VERSIONS)

    # Each article-source pair must have unique version records
    index_cols = ["key", "source", "dataset"]
    duplicate_mask = temp.duplicated(
        subset=index_cols + ["text_column"], keep=False
    )

    if duplicate_mask.any():
        raise ValueError(
            f"Found {duplicate_mask.sum()} duplicate article-version rows"
        )

    # Pivot to one row per article-source pair
    paired = temp.pivot(
        index=index_cols,
        columns="text_column",
        values="text"
    ).reset_index()

    paired.columns.name = None

    # Keep only records with complete, non-empty text
    text_cols = list(AUDIT_VERSIONS.values())

    # Remove missing values
    paired = paired.dropna(subset=text_cols).copy()

    # Remove empty or whitespace-only text
    valid_text = paired[text_cols].apply(
        lambda col: col.astype(str).str.strip().ne("")
    ).all(axis=1)

    paired = paired.loc[valid_text].copy()

    # Use a consistent column order
    paired = paired[index_cols + text_cols]
    paired = paired.reset_index(drop=True)

    return paired



from pathlib import Path

def validate_and_save_paired(
    paired,
    output_path="data/processed/paired_all.pkl"
):
    """Validate the aligned data and save it for later analysis."""
    id_cols = ["dataset", "key", "source"]
    text_cols = list(AUDIT_VERSIONS.values())

    # Check unique article-source pairs
    assert not paired.duplicated(id_cols).any()

    # Check all required text fields
    assert not paired[text_cols].isna().any().any()
    assert paired[text_cols].apply(
        lambda col: col.astype(str).str.strip().ne("")
    ).all().all()

    # Check dataset and source coverage
    assert set(paired["dataset"]) == set(DATASETS)
    assert paired["source"].nunique() == 7

    print("Paired shape:", paired.shape)

    # Display two representative rows per dataset
    samples = paired.groupby("dataset").head(2).copy()

    # Shorten text only for display
    for col in text_cols:
        samples[col] = samples[col].str.slice(0, 80)

    print("\nTwo sample rows per dataset:")
    print(samples.to_string(index=False))

    # Save the full dataset (not the shortened samples)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_pickle(path)

    print(f"\nSaved to: {path}")
