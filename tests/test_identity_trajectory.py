import numpy as np
import pandas as pd

from src.identity_trajectory import (
    aggregate_group_stage,
    make_group_split,
    summarize_separation,
)


def toy_features():
    rows = []
    for group_id, dataset, source_group in [
        ("a", "cmv", "Human"),
        ("b", "cmv", "LLM"),
        ("c", "xsum", "Human"),
        ("d", "xsum", "LLM"),
    ]:
        for stage_index, stage in enumerate(["t0", "t1", "t2", "t3"]):
            rows.append({
                "dataset": dataset,
                "group_id": group_id,
                "source_group": source_group,
                "stage": stage,
                "f1": float(stage_index + (source_group == "LLM")),
                "f2": float(stage_index),
            })
    return pd.DataFrame(rows)


def test_group_split_has_no_group_overlap_and_preserves_dataset_strata():
    frame = toy_features()
    train_ids, test_ids, assignments = make_group_split(
        frame, test_size=0.5, random_state=7
    )

    assert train_ids.isdisjoint(test_ids)
    assert set(assignments["split"]) == {"train", "test"}
    assert assignments.groupby(["dataset", "split"]).size().min() > 0


def test_aggregation_returns_one_row_per_group_source_stage():
    frame = pd.concat([toy_features(), toy_features()], ignore_index=True)
    aggregated = aggregate_group_stage(frame, ["f1", "f2"])

    assert len(aggregated) == 16
    assert not aggregated.duplicated(
        ["dataset", "group_id", "source_group", "stage"]
    ).any()


def test_separation_reports_stagewise_centroid_distance():
    frame = toy_features()
    frame["pc1"] = frame["f1"]
    frame["pc2"] = frame["f2"]
    summary = summarize_separation(frame, ["pc1", "pc2"])

    assert list(summary["stage"]) == ["t0", "t1", "t2", "t3"]
    assert np.isfinite(summary["centroid_distance"]).all()
