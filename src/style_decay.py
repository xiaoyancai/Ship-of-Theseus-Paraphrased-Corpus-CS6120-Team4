"""Analyze initial stylistic features and T0-relative retention from T0 to T3."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IDS = ['dataset', 'key', 'source', 'paraphraser']
KEYS = IDS + ['generation']
MODELS = ['ChatGPT', 'PaLM', 'Dipper', 'Pegasus']
FEATURES = ['type_token_ratio', 'sentence_length_mean', 'sentence_length_variance', 'punctuation_frequency']
LEXICAL = ['bleu', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1']
SEMANTIC = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
LABELS = {'type_token_ratio': 'Type–Token Ratio', 'sentence_length_mean': 'Mean sentence length',
          'sentence_length_variance': 'Sentence length variance', 'punctuation_frequency': 'Punctuation frequency',
          'bleu': 'BLEU', 'rouge1_f1': 'ROUGE-1 F1', 'rouge2_f1': 'ROUGE-2 F1',
          'rougeL_f1': 'ROUGE-L F1', 'bertscore_f1': 'BERTScore F1'}


def words(text):
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text.lower(), flags=re.UNICODE)


def stylistic_features(text):
    """Use lowercase word tokens, simple sentence boundaries, and Unicode punctuation."""
    tokens = words(text)
    if not tokens:
        raise ValueError('Text must contain at least one word token')
    sentences = re.split(r'[.!?]+(?:[\"\'”’)]*)\s+|[\r\n]+', text.strip())
    lengths = [len(words(sentence)) for sentence in sentences if words(sentence)]
    return dict(zip(FEATURES, [len(set(tokens)) / len(tokens), float(np.mean(lengths)),
                              float(np.var(lengths, ddof=0)),
                              100 * sum(unicodedata.category(c).startswith('P') for c in text) / len(tokens)]))


def relative_values(original, value):
    """Return signed relative change and bounded feature retention; zero T0 is explicit."""
    change = (value - original) / original if original != 0 else (0.0 if value == 0 else np.nan)
    denominator = abs(original) + abs(value)
    retention = 1 - abs(value - original) / denominator if denominator else 1.0
    return change, retention


def validate(frame):
    required = IDS + ['t0', 't1', 't2', 't3']
    if not set(required).issubset(frame):
        raise ValueError('Processed data is missing required columns')
    if frame[required].isna().any().any() or frame.duplicated(IDS).any():
        raise ValueError('Processed data contains missing values or duplicate chains')
    if set(frame.source) != {'Human'} or set(frame.paraphraser) != set(MODELS):
        raise ValueError('Expected Human sources and all four canonical paraphrasers')
    for _, group in frame.groupby(['dataset', 'key', 'source']):
        if group.t0.nunique() != 1:
            raise ValueError('T0 differs across paraphrasers for the same source document')


def load_scores(path, expected, metrics):
    scores = pd.read_csv(path)
    if not set(KEYS + metrics).issubset(scores) or scores.duplicated(KEYS).any():
        raise ValueError(f'Invalid score schema or duplicate keys: {path}')
    merged = expected.merge(scores[KEYS + metrics], on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError(f'Score coverage does not match complete chains: {path}')
    values = merged[metrics].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1e-6).any() or (values > 1 + 1e-6).any():
        raise ValueError(f'Missing or out-of-range scores: {path}')
    return merged.drop(columns='_merge')


def build_details(frame, lexical_path, bertscore_path):
    validate(frame)
    rows = []
    for chain in frame.to_dict('records'):
        original = stylistic_features(chain['t0'])
        for generation in range(4):
            row = {k: chain[k] for k in IDS}
            row['generation'] = f't{generation}'
            values = stylistic_features(chain[f't{generation}'])
            row.update(values)
            for feature in FEATURES:
                change, retention = relative_values(original[feature], values[feature])
                row[feature + '_relative_change'] = change
                row[feature + '_retention'] = retention
            row['style_retention'] = np.mean([row[f + '_retention'] for f in FEATURES])
            rows.append(row)
    details = pd.DataFrame(rows)
    expected = details.loc[details.generation != 't0', KEYS]
    lexical = load_scores(lexical_path, expected, LEXICAL)
    semantic = load_scores(bertscore_path, expected, SEMANTIC)
    scores = lexical.merge(semantic, on=KEYS, validate='one_to_one')
    details = details.merge(scores, on=KEYS, how='left', validate='one_to_one')
    # Identical T0/reference pairs have theoretical self-similarity 1 on these scales.
    details.loc[details.generation == 't0', LEXICAL + SEMANTIC] = 1.0
    for metric in LEXICAL + SEMANTIC:
        details[metric + '_retention'] = details[metric]
        details[metric + '_relative_change'] = details[metric] - 1.0
    coverage = frame.groupby(['dataset', 'key', 'source']).paraphraser.nunique()
    common = coverage[coverage == len(MODELS)].reset_index()[['dataset', 'key', 'source']]
    common['matched_cohort'] = True
    details = details.merge(common, how='left', on=['dataset', 'key', 'source'], validate='many_to_one')
    details['matched_cohort'] = details.matched_cohort.eq(True)
    return details


def summarize(details):
    metrics = [c for c in details if c not in KEYS + ['matched_cohort']]
    result = details.groupby(['dataset', 'paraphraser', 'generation'])[metrics].agg(['mean', 'std', 'count'])
    result.columns = ['_'.join(c) for c in result.columns]
    return result.reset_index()


def plot_curves(details, output, metrics, ylabel, title):
    datasets = sorted(details.dataset.unique())
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(4 * len(metrics), 3.5 * len(datasets)), squeeze=False)
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for model in MODELS:
                selected = details[(details.dataset == dataset) & (details.paraphraser == model)]
                means = selected.groupby('generation')[metric].mean().reindex(['t0', 't1', 't2', 't3'])
                ax.plot(range(4), means, marker='o', markersize=4, label='PaLM2 (PaLM)' if model == 'PaLM' else model)
            ax.set_title(dataset.upper() + ' — ' + LABELS.get(metric.replace('_retention', ''), metric.replace('_retention', '').title()))
            ax.set_xticks(range(4), ['T0', 'T1', 'T2', 'T3'])
            ax.set_ylabel(ylabel)
            if 'retention' in metric or metric in LEXICAL + SEMANTIC:
                ax.set_ylim(0, 1.04)
            ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, .06, 1, .95])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table(frame):
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def write_report(details, path):
    matched = details[details.matched_cohort]
    selected = ['bleu', 'rougeL_f1', 'bertscore_f1', 'style_retention']
    table = matched[matched.generation == 't3'].groupby(['dataset', 'paraphraser'])[selected].mean().reset_index()
    counts = details[details.generation == 't0'].groupby(['dataset', 'paraphraser']).size().reset_index(name='complete_chains')
    common_counts = matched[matched.generation == 't0'].groupby('dataset').key.nunique().to_dict()
    lines = ['# Project Update 1 — Style vs. Content Decay', '',
    'This analysis extends the existing similarity baseline with initial stylistic features. We compare complete Human T0 → T3 chains from Yelp and XSum for ChatGPT, PaLM, Dipper, and Pegasus(full). PaLM is displayed as PaLM2 (PaLM) in the figures to connect the project terminology with the stored corpus label; this analysis does not independently verify the underlying model version.', '',
    '## 1. Analysis Scope', '', markdown_table(counts), '',
    'All complete chains are retained in the main outputs. Because chain availability differs across paraphrasers, comparisons are also repeated on source documents available for all four paraphrasers: ' + ', '.join(f'{k.upper()} = {v}' for k, v in sorted(common_counts.items())) + '. Each matched source has the same T0 across models. Datasets are analyzed separately.', '',
    '## 2. Initial Stylistic Features', '',
    '- Type–Token Ratio: unique lowercase word tokens divided by all word tokens. Apostrophes within words are retained; numbers are excluded.',
    '- Mean sentence length: mean word-token count across nonempty sentences.',
    '- Sentence length variance: population variance of sentence word-token counts (ddof = 0); a single sentence has variance 0.',
    '- Punctuation frequency: Unicode punctuation characters per 100 word tokens.', '',
    'Sentence boundaries use a lightweight rule based on sentence-final punctuation followed by whitespace, or line breaks. Abbreviations and irregular punctuation can affect segmentation. TTR is sensitive to text length. These features provide initial descriptive evidence, not a validated measure of authorial identity.', '',
    '## 3. T0-relative Metrics', '',
    'BLEU and ROUGE-1/2/L F1 represent lexical retention. BERTScore precision/recall/F1 represent semantic similarity, with F1 used in the main figures. The supplied document-level score files are reused after exact key and coverage validation. The existing baseline defaults to distilbert-base-uncased without baseline rescaling; the supplied CSV files do not record the actual model, configuration, or package versions used, so their generation settings cannot be independently confirmed.', '',
    'For similarity metrics, T0 self-similarity is set to its theoretical value of 1. Retention is S(Tg,T0) / S(T0,T0), which equals the stored score; relative change is retention − 1. T0 anchors are not newly computed BERTScore observations.', '',
    'For each stylistic feature f, signed relative change is (f(Tg) − f(T0)) / f(T0). If both values are zero, change is 0; if only T0 is zero, the ratio is undefined and saved as a missing value. Per-metric counts in the summary expose these exclusions.', '',
    'To include zero-baseline features in a bounded descriptive curve, feature retention is 1 − |f(Tg) − f(T0)| / (|f(Tg)| + |f(T0)|), with retention 1 when both are zero. Style retention is the unweighted mean of the four feature retentions for each chain and generation. This is an exploratory feature-preservation index. Normalization gives a common T0 anchor, but does not make BLEU, BERTScore, and style retention empirically calibrated or directly equivalent.', '',
    '## 4. Decay Curves', '',
    '![Lexical retention](../figures/update1_lexical_decay.png)', '',
    '**Figure 1.** Mean BLEU and ROUGE-1/2/L F1 relative to the original T0 text. Each row is a dataset and each line is a paraphraser. Lower values indicate less surface overlap; the same complete chains contribute at every generation within each line.', '',
    '![Stylistic feature retention](../figures/update1_style_decay.png)', '',
    '**Figure 2.** T0-relative retention of each initial stylistic feature. A lower value means the feature has moved farther from its source value in either direction. Recovery is possible, so these trajectories should not be assumed to decay monotonically.', '',
    '![Retention comparison](../figures/update1_retention_comparison.png)', '',
    '**Figure 3.** BLEU, ROUGE-L F1, BERTScore F1, and exploratory style retention for all complete chains. Vertical differences across metric families are descriptive because their scales are not calibrated.', '',
    '![Matched comparison](../figures/update1_matched_comparison.png)', '',
    '**Figure 4.** The same comparison restricted to source documents with complete chains for all four paraphrasers. This controls source composition within each dataset. Lines show means; uncertainty is not represented by confidence bands.', '',
    '## 5. Results', '', 'T3 means on the matched cohort:', '', markdown_table(table), '']
    for dataset in sorted(matched.dataset.unique()):
        part = matched[matched.dataset == dataset]
        lines += [f'### {dataset.upper()}', '']
        for model in MODELS:
            means = part[part.paraphraser == model].groupby('generation')[selected].mean()
            a, b = means.loc['t1'], means.loc['t3']
            lines += [f'{model}: from T1 to T3, BLEU changes from {a.bleu:.3f} to {b.bleu:.3f}, ROUGE-L F1 from {a.rougeL_f1:.3f} to {b.rougeL_f1:.3f}, BERTScore F1 from {a.bertscore_f1:.3f} to {b.bertscore_f1:.3f}, and exploratory style retention from {a.style_retention:.3f} to {b.style_retention:.3f}.', '']
        ranking = table[table.dataset == dataset].sort_values('bertscore_f1', ascending=False)
        lines += ['At T3, BERTScore F1 ranks ' + ' > '.join(ranking.paraphraser) + ' on the matched sources. This describes this metric and sample, rather than overall paraphraser quality.', '']
    lines += ['### Initial Stylistic Patterns', '']
    for dataset in sorted(details.dataset.unique()):
        final = details[(details.dataset == dataset) & (details.generation == 't3')]
        means = final.groupby('paraphraser')[[f + '_retention' for f in FEATURES]].mean()
        lines += [f'{dataset.upper()}: across the four paraphrasers at T3, TTR retention ranges from {means.type_token_ratio_retention.min():.3f} to {means.type_token_ratio_retention.max():.3f}, mean sentence length retention from {means.sentence_length_mean_retention.min():.3f} to {means.sentence_length_mean_retention.max():.3f}, sentence length variance retention from {means.sentence_length_variance_retention.min():.3f} to {means.sentence_length_variance_retention.max():.3f}, and punctuation retention from {means.punctuation_frequency_retention.min():.3f} to {means.punctuation_frequency_retention.max():.3f}. These values use all complete chains, as in Figure 2.', '']
    lines += ['Sentence length variance shows the largest departure under this feature-distance definition, while TTR stays closest to its source value. Variance is sensitive to sentence segmentation and zero baselines, so this difference requires follow-up before being interpreted as a robust stylistic effect.', '']
    semantic_above = (table.bertscore_f1 > table.rougeL_f1).sum()
    style_above = (table.bertscore_f1 > table.style_retention).sum()
    lines += ['## 6. Initial Interpretation', '',
    f'At T3, BERTScore F1 is numerically higher than ROUGE-L F1 in {semantic_above} of {len(table)} matched dataset/paraphraser groups and higher than the exploratory style index in {style_above} of {len(table)} groups. The curves are consistent with substantial surface rewriting while semantic similarity remains comparatively high. This supports a preliminary hypothesis that semantic content may be more stable than lexical form under these metrics.', '',
    'The stronger claim that semantics lasts longer than style is not established. Raw BERTScore has a different score distribution from lexical overlap, and the style index depends on a chosen distance formula and four equally weighted features. T0 normalization alone cannot resolve these differences. A high style index can also conceal substantial changes in features not measured here. Decline relative to T0 is evidence of change, not proof of a loss of authorial identity.', '',
    '## 7. Limitations and Next Steps', '',
    '- Treat the stylistic features as an initial analysis. Add length-controlled lexical diversity, richer syntactic features, and authorship attribution before making claims about identity loss.',
    '- Validate semantic retention using human judgments or controlled meaning-change examples, and record the BERTScore model and configuration when regenerating scores.',
    '- Use source-level paired bootstrap intervals for model comparisons; the present means and standard deviations do not establish statistical significance.',
    '- Inspect signed feature changes alongside bounded retention. An increase in sentence length or TTR is still a departure from T0, and zero baselines require explicit handling.',
    '- Retain the matched-source sensitivity analysis because complete-chain counts differ across models, especially for XSum PaLM.', '']
    path.write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root
    results = root / 'data/results'
    figures = root / 'figures'
    figures.mkdir(exist_ok=True, parents=True)
    frame = pd.read_csv(root / 'data/processed/update1_processed.csv')
    details = build_details(frame, results / 'update1_lexical_scores.csv', results / 'update1_bertscore_scores.csv')
    details.to_csv(results / 'update1_style_decay_scores.csv', index=False)
    summarize(details).to_csv(results / 'update1_style_decay_summary.csv', index=False)
    matched = details[details.matched_cohort]
    summarize(matched).to_csv(results / 'update1_style_decay_matched_summary.csv', index=False)
    plot_curves(details, figures / 'update1_lexical_decay.png', LEXICAL, 'T0-relative retention', 'Lexical retention — all complete chains')
    plot_curves(details, figures / 'update1_style_decay.png', [f + '_retention' for f in FEATURES], 'T0-relative feature retention', 'Initial stylistic features — all complete chains')
    comparison = ['bleu', 'rougeL_f1', 'bertscore_f1', 'style_retention']
    plot_curves(details, figures / 'update1_retention_comparison.png', comparison, 'T0-relative retention', 'Style vs. content — descriptive, uncalibrated scales')
    plot_curves(matched, figures / 'update1_matched_comparison.png', comparison, 'T0-relative retention', 'Style vs. content — matched source documents')
    write_report(details, root / 'docs/update1_style_decay.md')
    (results / 'update1_style_decay_metadata.json').write_text(json.dumps({
        'chains': len(frame), 'generation_rows': len(details), 'matched_chains': int(matched.generation.eq('t0').sum()),
        'similarity_provenance': 'Supplied score CSVs; original runtime/model configuration not recorded.',
        't0_similarity': 'Theoretical self-similarity anchor 1; not recomputed.',
        'feature_variance_ddof': 0, 'summary_std_ddof': 1,
        'undefined_relative_changes': {f: int(details[f + '_relative_change'].isna().sum()) for f in FEATURES},
    }, indent=2) + '\n')
    print(f'Analyzed {len(frame):,} chains / {len(details):,} generation rows. Report: docs/update1_style_decay.md')


if __name__ == '__main__':
    main()
