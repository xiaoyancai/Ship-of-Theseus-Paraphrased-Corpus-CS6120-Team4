import pandas as pd

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