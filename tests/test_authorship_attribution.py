import numpy as np
import pandas as pd

from src.authorship_attribution import (
    FEATURES,
    identify_identity_loss,
    make_group_split,
    metrics_from_counts,
    unique_t0_origins,
)


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for dataset in ["alpha", "beta"]:
        for key_index in range(10):
            group_id = f"{dataset}::{key_index}"
            for source, source_group in [("Human", "Human"), ("Model", "LLM")]:
                for paraphraser in ["A", "B"]:
                    for stage in ["t0", "t1", "t2", "t3"]:
                        row = {
                            "dataset": dataset,
                            "key": str(key_index),
                            "source": source,
                            "source_group": source_group,
                            "paraphraser": paraphraser,
                            "group_id": group_id,
                            "stage": stage,
                        }
                        row.update({name: float(key_index) for name in FEATURES})
                        rows.append(row)
    return pd.DataFrame(rows)


def test_group_split_has_no_leakage_and_keeps_dataset_balance():
    frame = synthetic_frame()
    train, test, assignments = make_group_split(
        frame, test_size=0.20, random_state=7
    )
    assert not train.intersection(test)
    assert set(assignments.group_id) == train.union(test)
    assert assignments.groupby(["dataset", "split"]).size().to_dict() == {
        ("alpha", "test"): 2,
        ("alpha", "train"): 8,
        ("beta", "test"): 2,
        ("beta", "train"): 8,
    }


def test_unique_t0_origins_removes_paraphraser_duplicates():
    result = unique_t0_origins(synthetic_frame())
    assert len(result) == 40
    assert result["stage"].eq("t0").all()
    assert not result.duplicated(["dataset", "key", "source"]).any()


def test_binary_metrics_use_human_and_llm_symmetrically():
    metrics = metrics_from_counts(np.array([8, 2, 1, 9]))
    assert np.isclose(metrics["accuracy"], 0.85)
    assert np.isclose(metrics["recall_human"], 0.8)
    assert np.isclose(metrics["recall_llm"], 0.9)
    assert np.isclose(metrics["balanced_accuracy"], 0.85)
    assert np.isclose(metrics["recall_macro"], 0.85)


def test_identity_loss_does_not_overclaim_when_ci_is_above_chance():
    metrics = pd.DataFrame({
        "stage": ["T0", "T1", "T2", "T3"],
        "f1_macro": [0.80, 0.60, 0.58, 0.57],
        "balanced_accuracy_ci_high": [0.85, 0.70, 0.66, 0.64],
    })
    result = identify_identity_loss(metrics)
    assert result["largest_drop_stage"] == "T1"
    assert result["undetectable_stage"] is None
