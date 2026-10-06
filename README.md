# Ship of Theseus: Computational Forensics

This project investigates how iterative paraphrasing changes lexical form,
semantic content, and writing style. It uses the *Ship of Theseus Paraphrased
Corpus* to track texts from their original form (T0) through three successive
paraphrasing iterations (T1–T3).

This repository currently contains the work completed for **Project Update 1
(Week 6)**: data preprocessing, similarity baselines, an initial stylometric
audit, and T0–T3 decay visualizations.

## Team Members

- Chuwei Cai
- Xiaoyan Cai
- Chuwei Du
- Lexin Yi

## Research Questions

### RQ1 — Style vs. Content Decay

Which linguistic “planks” are replaced first? We compare changes in lexical
overlap, semantic similarity, and lightweight stylistic features from T0 to T3.

### RQ2 — Point of No Return

At what iteration does a text lose its original authorial identity? This will be
studied later by training an authorship-attribution model on T0 and evaluating
it on T1–T3.

### RQ3 — Paraphraser Fingerprints

Do ChatGPT, PaLM, Pegasus, and Dipper leave distinguishable linguistic traces?
This will be studied later with paraphraser classification and feature analysis.

## Project Update 1 Scope

Update 1 provides an initial investigation of RQ1.

- Datasets: Yelp and XSum
- Source: human-authored texts only (`source == "Human"`)
- Paraphrasers: ChatGPT, PaLM, Pegasus(full), and Dipper
- Generations: T0, T1, T2, and T3
- Inclusion rule: only complete T0–T3 chains are retained
- Lexical metrics: sentence BLEU and ROUGE-1/2/L F1
- Semantic metric: BERTScore precision, recall, and F1
- Initial style features:
  - Type–Token Ratio
  - mean sentence length
  - sentence-length variance
  - punctuation frequency
  - average word length

RQ2 and RQ3 are not claimed as completed in Update 1.

## Dataset

The project uses the public
[Ship of Theseus Paraphrased Corpus](https://github.com/tripto03/Ship_of_theseus_paraphrased_copus).
The full corpus contains seven datasets and four paraphrasers. Update 1 uses the
Yelp and XSum subsets.

Raw corpus files are not committed to this repository. To reproduce
preprocessing, place the required files at:

```text
data/raw/paraphrased_datasets/yelp_paraphrased.csv
data/raw/paraphrased_datasets/xsum_paraphrased.csv
```

The raw files contain the following fields:

```text
source, key, text, version_name
```

## Preprocessing

The preprocessing pipeline:

1. keeps rows with `source == "Human"`;
2. maps raw version names onto T0, T1, T2, and T3;
3. constructs separate chains for ChatGPT, PaLM, Pegasus(full), and Dipper;
4. removes chains missing any generation;
5. checks missing values and duplicate document/paraphraser rows; and
6. writes a shared processed file for downstream experiments.

The primary processed file is:

```text
data/processed/update1_processed.csv
```

Its columns are:

```text
dataset, key, source, paraphraser, t0, t1, t2, t3
```

### Complete T0–T3 Chains

| Dataset | ChatGPT | PaLM | Pegasus | Dipper |
|---|---:|---:|---:|---:|
| Yelp | 488 | 486 | 491 | 491 |
| XSum | 471 | 340 | 475 | 476 |

The processed data contains **3,718 complete chains**.

## Repository Structure

```text
.
├── data/
│   ├── raw/          # Local raw corpus files; not committed
│   ├── processed/    # Standardized T0–T3 chains
│   ├── metadata/     # Dataset and chain-count tables
│   └── results/      # Document-level and aggregate results
├── docs/             # Generated analysis and interpretation
├── figures/          # Update 1 plots
├── notebooks/        # Data exploration and analysis notebooks
├── src/              # Reusable preprocessing and analysis code
├── tests/            # Automated checks
└── requirements.txt  # Python dependencies
```

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Reproduce the Update 1 Analysis

Run all commands from the repository root.

### 1. Preprocess and inspect the data

Open and run:

```text
notebooks/01_data_exploration.ipynb
```

The already processed Update 1 CSV files are included in the repository, so the
remaining experiments can be reproduced without rerunning raw-data ingestion.

### 2. Compute lexical similarity

```bash
python src/similarity_baseline.py --metrics lexical \
  --details-output data/results/update1_lexical_scores.csv \
  --summary-output data/results/update1_lexical_summary.csv
```

### 3. Compute BERTScore

```bash
python src/similarity_baseline.py --metrics bertscore \
  --details-output data/results/update1_bertscore_scores.csv \
  --summary-output data/results/update1_bertscore_summary.csv
```

The default BERTScore configuration uses `distilbert-base-uncased`, English,
no baseline rescaling, batch size 8, and CPU execution.

### 4. Generate style analysis and figures

```bash
python src/style_decay.py
```

Alternatively, open and run:

```text
notebooks/02_style_content_decay.ipynb
```

This analysis validates exact document/generation alignment before combining
lexical, semantic, and stylistic results.

### 5. Run the tests

```bash
python -m pytest -q
```

## Update 1 Outputs

### Similarity results

- `data/results/update1_lexical_scores.csv`
- `data/results/update1_lexical_summary.csv`
- `data/results/update1_bertscore_scores.csv`
- `data/results/update1_bertscore_summary.csv`

### Style-decay results

- `data/results/update1_style_decay_scores.csv`
- `data/results/update1_style_decay_summary.csv`
- `data/results/update1_style_decay_matched_summary.csv`
- `data/results/update1_style_decay_metadata.json`

### Figures and interpretation

- `figures/update1_lexical_decay.png`
- `figures/update1_style_decay.png`
- `figures/update1_retention_comparison.png`
- `figures/update1_matched_comparison.png`
- `docs/update1_style_decay.md`

## Initial Results

- Lexical overlap generally decreases from T1 to T3.
- BERTScore remains comparatively high while BLEU and ROUGE decline, which is
  consistent with meaning being more stable than surface wording.
- Sentence-length variance shows the largest departure among the five initial
  style features, while Type–Token Ratio and average word length remain closer
  to their T0 values.
- Results vary across datasets and paraphrasers.
- A matched-source analysis is included because complete-chain counts differ
  across paraphrasers, especially for XSum PaLM.

These are descriptive initial findings. BLEU, BERTScore, and the exploratory
style-retention index have different score distributions and should not be
treated as directly equivalent scales.

## Limitations

- Update 1 analyzes only the Yelp and XSum human-source subsets.
- The current style analysis uses five lightweight surface features.
- POS distributions and dependency-based syntactic features are not yet
  included.
- Current plots show group means without paired confidence intervals.
- Linguistic change relative to T0 does not by itself prove loss of authorial
  identity.

## Plan for Next

In the coming weeks, the team plans to:

1. add POS-frequency and dependency-based structural features to strengthen
   the RQ1 analysis;
2. add source-level paired confidence intervals and significance checks;
3. train an authorship-attribution model on T0 only and evaluate accuracy and
   F1 on T1, T2, and T3 for RQ2;
4. identify a possible “point of no return” through performance-decay curves
   and error analysis;
5. train a multi-class paraphraser-identification model for RQ3 and report a
   confusion matrix and feature importance; and
6. extend the experiments to additional corpus subsets after validating the
   Yelp/XSum pipeline.

The project uses the public
[Ship of Theseus Paraphrased Corpus](https://github.com/tripto03/Ship_of_theseus_paraphrased_copus).