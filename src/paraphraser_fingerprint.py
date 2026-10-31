"""RQ3: group-held-out initial (T1) paraphraser classification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.authorship_attribution import FEATURES, make_group_split

LABELS = ['ChatGPT', 'PaLM', 'Dipper', 'Pegasus']
MAPPING = {'ChatGPT': 'ChatGPT', 'PaLM': 'PaLM', 'Dipper': 'Dipper', 'Pegasus(full)': 'Pegasus'}


def prepare_data(frame):
    required = {'dataset', 'key', 'source', 'source_group', 'group_id', 'stage', 'paraphraser', *FEATURES}
    if required - set(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    data = frame.loc[frame.stage.str.lower().eq('t1') & frame.paraphraser.isin(MAPPING)].copy()
    if data.empty or data[list(required - set(FEATURES))].isna().any().any():
        raise ValueError('Empty T1 data or missing identifiers')
    data['label'] = data.paraphraser.map(MAPPING)
    if set(data.label) != set(LABELS):
        raise ValueError('All four target classes are required')
    if data.duplicated(['dataset', 'key', 'source', 'paraphraser']).any():
        raise ValueError('Duplicate T1 document/paraphraser rows')
    if data.groupby(['dataset', 'key']).group_id.nunique().gt(1).any():
        raise ValueError('A document key maps to multiple group_id values')
    if data.groupby('group_id').dataset.nunique().gt(1).any():
        raise ValueError('A group_id maps to multiple datasets')
    values = data[FEATURES].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('Infinite feature values')
    return data


def score(y, pred):
    return {'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, labels=LABELS, average='macro', zero_division=0)}


def run(input_path, output_dir, figure_dir, seed=42):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(pd.read_pickle(input_path))
    train_ids, test_ids, assignments = make_group_split(data, random_state=seed)
    train = data[data.group_id.isin(train_ids)]
    test = data[data.group_id.isin(test_ids)]
    assert not set(train.group_id) & set(test.group_id)
    if set(train.label) != set(LABELS) or set(test.label) != set(LABELS):
        raise ValueError('Split must contain all classes')
    assignments.to_csv(output_dir / 'rq3_group_split.csv', index=False)
    models = {
        'majority_baseline': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)),
        'random_forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight='balanced', random_state=seed, n_jobs=-1)),
    }
    metrics, subgroup, counts, reports = [], [], [], []
    predictions = test[['dataset', 'key', 'source', 'source_group', 'group_id', 'paraphraser', 'label']].copy()
    for name, model in models.items():
        print(f'Training {name}: {len(train):,} train / {len(test):,} test', flush=True)
        model.fit(train[FEATURES], train.label)
        pred = model.predict(test[FEATURES])
        predictions[name] = pred
        metrics.append({'model': name, 'n_train': len(train), 'n_test': len(test), **score(test.label, pred)})
        report = pd.DataFrame(classification_report(test.label, pred, labels=LABELS, output_dict=True, zero_division=0)).T.rename_axis('class').reset_index()
        report['model'] = name
        reports.append(report)
        for column in ['dataset', 'source_group']:
            for value, part in predictions.groupby(column):
                subgroup.append({'model': name, 'dimension': column, 'value': value, 'n': len(part), **score(part.label, part[name])})
        cm = confusion_matrix(test.label, pred, labels=LABELS)
        for i, true in enumerate(LABELS):
            for j, predicted in enumerate(LABELS):
                counts.append({'model': name, 'true_label': true, 'predicted_label': predicted, 'count': int(cm[i,j])})
        if name != 'majority_baseline':
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, normalize in zip(axes, [None, 'true']):
                ConfusionMatrixDisplay.from_predictions(test.label, pred, labels=LABELS, normalize=normalize, ax=ax, cmap='Blues', colorbar=False, values_format='d' if normalize is None else '.2f')
                ax.set_title('Counts' if normalize is None else 'Row-normalized (recall)')
            title = ('RQ3: Logistic Regression Confusion Matrices for T1 Paraphraser Classification (Group-Held-Out Test Set)'
                     if name == 'logistic_regression' else 'RQ3: Random Forest Confusion Matrices for T1 Paraphraser Classification (Group-Held-Out Test Set)')
            fig.suptitle(title, fontsize=10)
            fig.tight_layout()
            fig.savefig(figure_dir / f'rq3_{name}_confusion.png', dpi=180)
            plt.close(fig)
    # Coefficients describe conditional associations, not causal signatures.
    logistic = models['logistic_regression'][-1]
    coefficients = pd.DataFrame(logistic.coef_.T, index=FEATURES, columns=logistic.classes_)
    coefficients.rename_axis('feature').to_csv(output_dir / 'rq3_standardized_coefficients.csv')
    means = train.groupby('label')[FEATURES].mean().T
    means.rename_axis('feature').to_csv(output_dir / 'rq3_train_feature_means.csv')
    fig, ax = plt.subplots(figsize=(9, 10))
    matrix = coefficients[LABELS].to_numpy()
    limit = np.abs(matrix).max()
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(4), LABELS)
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.set_title('RQ3: standardized logistic coefficients (training groups)')
    fig.colorbar(im, ax=ax, label='Coefficient (positive favors class)')
    fig.tight_layout()
    fig.savefig(figure_dir / 'rq3_feature_coefficients.png', dpi=180)
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(output_dir / 'rq3_metrics.csv', index=False)
    pd.DataFrame(subgroup).to_csv(output_dir / 'rq3_subgroup_metrics.csv', index=False)
    pd.DataFrame(counts).to_csv(output_dir / 'rq3_confusion_matrices.csv', index=False)
    pd.concat(reports).to_csv(output_dir / 'rq3_classification_reports.csv', index=False)
    predictions.to_csv(output_dir / 'rq3_test_predictions.csv', index=False)
    data.groupby(['dataset', 'source_group', 'label']).size().rename('n').to_csv(output_dir / 'rq3_sample_counts.csv')
    summary = {'stage': 't1', 'labels': LABELS, 'pegasus_variant': 'Pegasus(full)', 'features': FEATURES, 'seed': seed, 'test_fraction': 0.2, 'n_train_groups': len(train_ids), 'n_test_groups': len(test_ids), 'group_overlap': len(train_ids & test_ids), 'n_train': len(train), 'n_test': len(test), 'metrics': metrics, 'missing_feature_values': int(data[FEATURES].isna().sum().sum()), 'input': str(input_path)}
    (output_dir / 'rq3_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/update2_features.pkl'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/results/update2'))
    parser.add_argument('--figure-dir', type=Path, default=Path('figures/update2'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.figure_dir, args.seed)
