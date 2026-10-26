"""Verify overlap definitions and zero-entity denominator handling."""
import math
from src.ner_stability import entity_metrics


def test_overlap():
    assert entity_metrics({'alice', 'paris'}, {'paris', 'london'}) == {
        'jaccard': 1 / 3, 'recall': .5, 'precision': .5}


def test_both_empty():
    scores = entity_metrics(set(), set())
    assert all(math.isnan(value) for value in scores.values())


def test_erasure():
    scores = entity_metrics({'alice'}, set())
    assert scores['jaccard'] == scores['recall'] == 0
    assert math.isnan(scores['precision'])


def test_addition_to_empty_original():
    scores = entity_metrics(set(), {'alice'})
    assert all(math.isnan(value) for value in scores.values())
