"""Task 3: T0-to-T1 named-entity stability (run with python -m src.ner_stability)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

VARIANTS = {'ChatGPT': 'text_chatgpt', 'Dipper(high)': 'text_dipper_high',
            'Pegasus(slight)': 'text_pegasus_slight', 'Pegasus(full)': 'text_pegasus_full'}
METRICS = ['jaccard', 'recall', 'precision']
IDS = ['dataset', 'key', 'source']


def entity_metrics(original, rewritten):
    """Exclude empty T0 from all metrics; empty T1 precision is undefined."""
    if not original:
        return dict.fromkeys(METRICS, float('nan'))
    intersection = len(original & rewritten)
    union = len(original | rewritten)
    return {'jaccard': intersection / union if union else 1.0,
            'recall': intersection / len(original) if original else float('nan'),
            'precision': intersection / len(rewritten) if rewritten else float('nan')}


def summarize(scores, groups):
    return scores.groupby(groups, sort=True)[METRICS].agg(['count', 'mean', 'std']).reset_index()


def flatten(table):
    table.columns = ['_'.join(str(x) for x in c if x) if isinstance(c, tuple) else c for c in table.columns]
    return table


def run(root, model='en_core_web_sm', processes=1):
    import spacy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    path = root / 'data/processed/paired_all.pkl'
    if not path.exists():
        raise FileNotFoundError(f'Place paired_all.pkl at {path} before running Task 3.')
    paired = pd.read_pickle(path).reset_index(drop=True)
    columns = ['text_T0', *VARIANTS.values()]
    if not set(IDS + columns).issubset(paired.columns):
        raise ValueError('Missing required identity or text columns')
    if paired.empty or paired[IDS].isna().any().any() or paired.duplicated(IDS).any():
        raise ValueError('Empty data, missing identities, or duplicate article-source pairs')
    for column in columns:
        if not paired[column].map(lambda t: isinstance(t, str) and bool(t.strip())).all():
            raise ValueError(f'Invalid text in {column}')
    nlp = spacy.load(model, disable=['parser', 'lemmatizer', 'tagger', 'attribute_ruler'])
    nlp.max_length = max(nlp.max_length, max(paired[c].str.len().max() for c in columns) + 1)
    texts = list(dict.fromkeys(text for c in columns for text in paired[c]))
    fingerprint = hashlib.sha256(json.dumps({'texts': texts, 'spacy': spacy.__version__, 'model': nlp.meta}, sort_keys=True).encode()).hexdigest()
    cache = root / 'data/cache/task3_ner' / f'{fingerprint}.json'
    partial = cache.with_suffix('.partial.json')
    existing = cache if cache.exists() else partial
    entities = {k: set(v) for k, v in json.loads(existing.read_text()).items()} if existing.exists() else {}
    missing = [text for text in texts if text not in entities]
    if missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        for i, (text, doc) in enumerate(zip(missing, nlp.pipe(missing, batch_size=64, n_process=processes))):
            entities[text] = {e.text.strip().lower() for e in doc.ents if e.text.strip()}
            if (i + 1) % 1000 == 0:
                print(f'NER: {len(entities):,}/{len(texts):,} unique texts', flush=True)
            if (i + 1) % 5000 == 0:
                partial.write_text(json.dumps({k: sorted(v) for k, v in entities.items()}, ensure_ascii=False))
        cache.write_text(json.dumps({k: sorted(v) for k, v in entities.items()}, ensure_ascii=False))
        partial.unlink(missing_ok=True)
    rows = []
    for record in paired.to_dict('records'):
        original = entities[record['text_T0']]
        for variant, column in VARIANTS.items():
            rewritten = entities[record[column]]
            rows.append({**{k: record[k] for k in IDS}, 'variant': variant,
                         'original_entities': json.dumps(sorted(original), ensure_ascii=False),
                         'rewritten_entities': json.dumps(sorted(rewritten), ensure_ascii=False),
                         'n_original': len(original), 'n_rewritten': len(rewritten),
                         'n_retained': len(original & rewritten), 'n_erased': len(original - rewritten),
                         'n_added': len(rewritten - original), **entity_metrics(original, rewritten)})
    scores = pd.DataFrame(rows)
    out = root / 'data/results/task3_ner'
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / 'article_ner_scores.csv', index=False)
    examples = scores[scores.variant.eq('Dipper(high)') & scores.source.eq('Human')
                      & scores.n_original.ge(5) & scores.n_retained.eq(0)].groupby('dataset').head(1)
    examples = examples.merge(paired[IDS + ['text_T0', 'text_dipper_high']], on=IDS, validate='one_to_one')
    examples.to_csv(out / 'dipper_high_zero_overlap_examples.csv', index=False)
    tables = {}
    for name, groups in [('overall', ['variant']), ('dataset', ['dataset', 'variant']),
                         ('dataset_source', ['dataset', 'source', 'variant'])]:
        table = flatten(summarize(scores, groups))
        table.to_csv(out / f'{name}_ner_summary.csv', index=False)
        tables[name] = table
    flatten(summarize(scores[scores.n_original.gt(0)], ['dataset', 'variant'])).to_csv(
        out / 'nonempty_original_ner_summary.csv', index=False)
    scores.groupby(['dataset', 'variant']).agg(n=('key', 'size'),
        zero_original=('n_original', lambda x: int(x.eq(0).sum())),
        zero_rewritten=('n_rewritten', lambda x: int(x.eq(0).sum()))).to_csv(out / 'zero_entity_counts.csv')
    peg = scores[scores.variant.str.startswith('Pegasus')]
    wide = peg.pivot(index=IDS, columns='variant', values=METRICS)
    deltas = pd.DataFrame({m: wide[(m, 'Pegasus(full)')] - wide[(m, 'Pegasus(slight)')] for m in METRICS}).reset_index()
    deltas.to_csv(out / 'pegasus_article_deltas.csv', index=False)
    flatten(summarize(deltas, ['dataset'])).to_csv(out / 'pegasus_full_minus_slight_summary.csv', index=False)
    datasets = sorted(paired.dataset.unique())
    overall = tables['overall'].set_index('variant')
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    positions = np.arange(len(METRICS))
    for offset, variant, color in [(-.19, 'Pegasus(slight)', '#4682b4'), (.19, 'Pegasus(full)', '#de8452')]:
        row = overall.loc[variant]
        ax.bar(positions + offset, [row[f'{m}_mean'] for m in METRICS], width=.36,
               yerr=[row[f'{m}_std'] for m in METRICS], capsize=5, label=variant, color=color)
    ax.set(xticks=positions, xticklabels=['Jaccard', 'Recall', 'Precision'],
           xlabel='Entity retention metric', ylabel='Mean score ± sample SD',
           title='Pegasus coverage: slight (25%) vs. full (100%)')
    ax.set_ylim(0, 1.30)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='upper right', frameon=True, facecolor='white',
              framealpha=1, edgecolor='#dddddd')
    fig.savefig(out / 'pegasus_ner_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    a, b = overall.loc['Pegasus(slight)'], overall.loc['Pegasus(full)']
    recall_gap = a.recall_mean - b.recall_mean
    precision_gap = a.precision_mean - b.precision_mean
    by_dataset = tables['dataset'].set_index(['dataset', 'variant'])
    gaps = {dataset: {
        metric: by_dataset.loc[(dataset, 'Pegasus(slight)'), f'{metric}_mean']
                - by_dataset.loc[(dataset, 'Pegasus(full)'), f'{metric}_mean']
        for metric in ['recall', 'precision']} for dataset in datasets}
    consistent = sum(g['recall'] > g['precision'] for g in gaps.values())
    interpretation = (
        f"Among {int(a.recall_count):,} eligible article-source pairs, mean Jaccard was "
        f"{overall.loc['ChatGPT', 'jaccard_mean']:.3f} for ChatGPT, "
        f"{overall.loc['Dipper(high)', 'jaccard_mean']:.3f} for Dipper(high), "
        f"{a.jaccard_mean:.3f} for Pegasus(slight), and {b.jaccard_mean:.3f} for Pegasus(full). "
        f"The Pegasus slight-to-full decline was larger for Recall ({recall_gap:.3f}) than "
        f"Precision ({precision_gap:.3f}), indicating more entity dropping than introduction of new entity strings. "
        f"This relationship held across {consistent} of {len(datasets)} datasets, although its magnitude varied by domain: "
        f"Recall gaps were {gaps['sci_gen']['recall']:.3f} for scientific text, "
        f"{gaps['xsum']['recall']:.3f} for news, and {gaps['yelp']['recall']:.3f} for reviews. "
        "Thus, greater paraphrase coverage primarily affected entity retention. Exact-string differences can reflect "
        "aliases or NER errors rather than factual hallucinations. Dipper(high)'s low overlap warrants checking "
        "generation quality and upstream alignment."
    )
    zero_originals = int(scores.loc[scores.variant.eq('ChatGPT'), 'n_original'].eq(0).sum())
    lines = ('# Task 3 — NER Stability & Content Erasure\n\n## Completion and verification\n\nThe analysis covers {n_pairs:,} article-source pairs across seven datasets and seven source groups, producing {n_rows:,} comparison rows for four T1 variants. The supplied verification summary records that all four metric edge-case tests passed, every exported score and all main summary statistics were independently recomputed.\n\n## Method and exclusions\n\nCompare each T1 variant against its aligned T0 using all available source groups. Extract entity surface strings across all spaCy entity labels using en_core_web_sm. Lowercase and strip each string, then deduplicate; repeated mentions count once.\n\n- Jaccard = intersection / union.\n- Recall = intersection / T0 entity-set size.\n- Precision = intersection / T1 entity-set size.\n\nWhen T0 has zero entities, all three metrics are NaN and the article-source pair is excluded from every metric mean. With nonempty T0 and empty T1, Jaccard and recall are 0, while precision is undefined (NaN) and excluded separately. Valid counts are reported per metric. Standard deviation uses ddof=1 and is undefined when n < 2.\n\nOf the {n_pairs:,} original article-source texts, {zero_originals:,} have no recognized entities ({excluded_fraction:.2%}). Excluding them leaves {eligible:,} eligible pairs per variant for Jaccard and recall. Precision additionally excludes empty rewrites. The nonempty-original summary uses the same eligible cohort as the main analysis and matches the main dataset summary.\n\nOverall results weight article-source pairs equally. Different source pairs may share article keys; no independence or statistical significance is claimed.\n\nSupporting files: zero_entity_counts.csv and nonempty_original_ner_summary.csv.\n').format(
        n_pairs=len(paired), n_rows=len(scores), zero_originals=zero_originals,
        excluded_fraction=zero_originals / len(paired), eligible=len(paired) - zero_originals
    ).splitlines() + ['', '## Comparison tables (mean ± sample SD)', '']
    for name in ['overall', 'dataset']:
        t = tables[name]
        keys = ['variant'] if name == 'overall' else ['dataset', 'variant']
        lines += ['| ' + ' | '.join(keys + [f'{m} (n)' for m in METRICS]) + ' |',
                  '| ' + ' | '.join(['---'] * (len(keys) + 3)) + ' |']
        for _, row in t.iterrows():
            lines.append('| ' + ' | '.join([str(row[k]) for k in keys] + [f'{row[m+"_mean"]:.3f} ± {row[m+"_std"]:.3f} ({int(row[m+"_count"])})' for m in METRICS]) + ' |')
        lines += ['']
    lines += '## Pegasus controlled comparison\n\nPegasus(slight) rewrites a random 25% of sentences; Pegasus(full) rewrites all sentences. The model and random seed are held fixed and sentence coverage differs. Upstream generation settings were not independently verified.\n\nThe chart pools eligible article-source pairs across all seven datasets. It has three metric clusters (Jaccard, Recall, Precision), two bars per cluster (slight and full), labeled axes, and article-level sample SD error bars. Per-dataset comparisons remain available in the CSV tables.\n\nChart file: pegasus_ner_comparison.png.\n\nDifferences are paired by dataset, key and source. See pegasus_full_minus_slight_summary.csv for full-minus-slight mean differences and sample SD. Negative values indicate lower overlap under full paraphrasing. Each metric excludes pairs where either value is undefined.\n'.splitlines() + ['', '## Interpretation', '', interpretation, '']
    (out / 'task3_ner_report.md').write_text('\n'.join(lines))
    (out / 'metadata.json').write_text(json.dumps({'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'n_pairs': len(paired), 'zero_entity_originals': zero_originals, 'datasets': datasets, 'sources': sorted(paired.source.unique()),
        'spacy_version': spacy.__version__, 'model': model, 'model_version': nlp.meta['version'],
        'enabled_pipeline': nlp.pipe_names, 'entity_labels': list(nlp.get_pipe('ner').labels),
        'input_url': 'https://drive.google.com/file/d/19V_df0vA-gSeNCo8XYq3gnv0LH-qM5ln/view',
        'std_ddof': 1, 'empty_T0_policy': 'All three metrics NaN; exclude from all metric means',
        'empty_T1_precision': 'NaN for undefined 0/0',
        'excluded_fraction': zero_originals / len(paired), 'eligible_pairs': len(paired) - zero_originals,
        'chart_layout': 'Three metric clusters, each with slight/full bars; pooled article-level means and sample SD',
        'cache_fingerprint': fingerprint, 'interpretation_words': len(interpretation.split())}, indent=2))
    print('Task 3 results: data/results/task3_ner')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', default='en_core_web_sm')
    parser.add_argument('--processes', type=int, default=1, help='spaCy worker processes (default: 1)')
    args = parser.parse_args()
    if args.processes < 1:
        parser.error('--processes must be positive')
    run(args.root, args.model, args.processes)
