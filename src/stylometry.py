
import numpy as np
import spacy
from src.style_decay import stylistic_features

UPOS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
    "INTJ", "NOUN", "NUM", "PART", "PRON",
    "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]


def pos_features(doc):
    """Return normalized proportions for all 17 POS tags."""
    tokens = [token for token in doc if not token.is_space]
    total = len(tokens)

    return {
        f"pos_{tag}": (
            sum(token.pos_ == tag for token in tokens) / total
            if total else 0.0
        )
        for tag in UPOS
    }


def dependency_depth(doc):
    """Return mean maximum dependency depth across sentences."""
    depths = []

    for sent in doc.sents:
        depth = max(
            (len(list(token.ancestors)) for token in sent),
            default=0
        )
        depths.append(depth)

    return float(np.mean(depths)) if depths else 0.0


def extract_features(doc):
    """Extract POS, dependency, and stylometric features."""
    features = pos_features(doc)
    features["dependency_depth"] = dependency_depth(doc)

    # Reuse the five stylistic features from Update 1
    features.update(stylistic_features(doc.text))

    return features


import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.style_decay import FEATURES as STYLE_FEATURES


def safe_extract_features(doc):
    """Handle texts without word tokens."""
    try:
        return extract_features(doc)
    except ValueError as error:
        if "at least one word token" not in str(error):
            raise

        features = pos_features(doc)
        features["dependency_depth"] = dependency_depth(doc)

        for name in STYLE_FEATURES:
            features[name] = float("nan")

        return features


def build_feature_dataset(limit_chains=None, chunk_size=1000):
    """Extract features with text deduplication and resumable caching."""
    root = Path(__file__).resolve().parent.parent
    input_path = root / "data/processed/update2_chains.pkl"

    chains = pd.read_pickle(input_path)

    if limit_chains is not None:
        chains = chains.head(limit_chains).copy()

    id_cols = [
        "dataset", "key", "source", "source_group",
        "paraphraser", "group_id"
    ]

    # Convert T0-T3 into separate rows.
    long_df = chains.melt(
        id_vars=id_cols,
        value_vars=["t0", "t1", "t2", "t3"],
        var_name="stage",
        value_name="text"
    )

    # Assign the same ID to identical texts.
    text_ids, unique_texts = pd.factorize(
        long_df["text"], sort=False
    )

    long_df["text_id"] = text_ids
    long_df = long_df.drop(columns="text")

    print("Total text versions:", len(long_df))
    print("Unique texts:", len(unique_texts))

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # Separate caches for different inputs and code versions.
    settings = {
        "input_size": input_path.stat().st_size,
        "input_modified": input_path.stat().st_mtime_ns,
        "limit_chains": limit_chains,
        "code_hash": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "spacy_version": spacy.__version__,
        "model_version": nlp.meta["version"],
        "feature_schema": 1,
    }

    cache_id = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()[:16]

    cache_dir = root / "data/cache/update2_stylometry" / cache_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_files = []

    for start in range(0, len(unique_texts), chunk_size):
        end = min(start + chunk_size, len(unique_texts))
        cache_file = cache_dir / f"part_{start:07d}.pkl"
        cache_files.append(cache_file)

        # Skip chunks already processed.
        if cache_file.exists():
            continue

        texts = unique_texts[start:end].tolist()
        records = []

        for offset, doc in enumerate(
            nlp.pipe(texts, batch_size=64)
        ):
            features = safe_extract_features(doc)
            features["text_id"] = start + offset
            records.append(features)

        part = pd.DataFrame(records)

        # Save atomically to avoid incomplete cache files.
        temporary = cache_file.with_name(
            cache_file.name + ".tmp"
        )
        part.to_pickle(temporary)
        temporary.replace(cache_file)

        print(f"Processed unique texts: {end:,}/{len(unique_texts):,}")

    # Combine cached feature chunks.
    extracted = pd.concat(
        [pd.read_pickle(path) for path in cache_files],
        ignore_index=True
    ).set_index("text_id")

    assert len(extracted) == len(unique_texts)
    assert extracted.index.is_unique

    features_df = long_df.join(extracted, on="text_id")
    features_df = features_df.drop(columns="text_id")

    assert len(features_df) == len(chains) * 4

    output_name = (
        "update2_features.pkl"
        if limit_chains is None
        else "update2_features_sample.pkl"
    )

    output = root / "data/processed" / output_name
    features_df.to_pickle(output)

    print("\nFeature shape:", features_df.shape)
    print("Missing TTR values:",
          features_df["type_token_ratio"].isna().sum())
    print("Saved to:", output)

    return features_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-chains", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)

    args = parser.parse_args()

    build_feature_dataset(
        limit_chains=args.limit_chains,
        chunk_size=args.chunk_size
    )

