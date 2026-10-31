import pandas as pd
import unittest
from math import isclose

def raises(error, match):
    return unittest.TestCase().assertRaisesRegex(error, match)

from src.authorship_attribution import FEATURES, make_group_split
from src.paraphraser_fingerprint import prepare_data, MAPPING, score


def fixture_data():
    return pd.DataFrame([
        {'dataset': dataset, 'key': key, 'source': source, 'source_group': 'Human' if source == 'Human' else 'LLM', 'group_id': f'{dataset}::{key}', 'stage': stage, 'paraphraser': label, **{feature: float(key + 1) for feature in FEATURES}}
        for dataset in ['a', 'b'] for key in range(10)
        for source in ['Human', 'LLM'] for label in MAPPING
        for stage in ['t0', 't1', 't2', 't3']
    ])


def test_split_keeps_sources_paraphrasers_and_stages_together():
    frame = fixture_data()
    data = prepare_data(frame)
    assert set(data.stage) == {'t1'}
    assert set(data.label) == {'ChatGPT', 'PaLM', 'Dipper', 'Pegasus'}
    train, test, assignments = make_group_split(data)
    assert not train & test
    assert train | test == set(frame.group_id)
    assert len(test) == 4
    merged = frame.merge(assignments[['group_id', 'split']], on='group_id')
    assert merged.groupby(['dataset', 'key'])['split'].nunique().eq(1).all()
    assert assignments.equals(make_group_split(data)[2])


def test_rejects_duplicate_and_inconsistent_groups():
    data = fixture_data()
    with raises(ValueError, match='Duplicate'):
        prepare_data(pd.concat([data, data]))
    data.loc[data.source.eq('LLM'), 'group_id'] += 'other'
    with raises(ValueError, match='multiple group_id'):
        prepare_data(data)


def test_macro_metric_uses_all_four_classes():
    result = score(['ChatGPT', 'PaLM', 'Dipper', 'Pegasus'], ['ChatGPT'] * 4)
    assert result['accuracy'] == .25
    assert isclose(result['macro_f1'], .1)

if __name__ == '__main__':
    test_split_keeps_sources_paraphrasers_and_stages_together()
    test_rejects_duplicate_and_inconsistent_groups()
    test_macro_metric_uses_all_four_classes()
    print('3 RQ3 checks passed')
