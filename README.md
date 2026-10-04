# Ship of Theseus: Computational Forensics

This repository investigates linguistic decay and authorial identity loss
under iterative paraphrasing by large language models.

## Team Members

- Chuwei Cai
- Xiaoyan Cai
- Chuwei Du
- Lexin Yi

## Research Questions

### RQ1 — Style vs. Content Decay
**Which linguistic “planks” are replaced first?**  
Investigate whether stylistic features decay faster than semantic content across iterative paraphrasing from T0 to T3.

### RQ2 — Point of No Return
**When does a text lose its authorial identity?**  
Examine how authorship attribution performance changes from T0 to T3 and determine whether identity loss occurs sharply or gradually.

### RQ3 — Paraphraser Fingerprints
**Do different paraphrasers leave distinct traces?**  
Determine whether ChatGPT, PaLM2, Pegasus, and Dipper produce distinguishable linguistic patterns that can identify the paraphrasing model.
## Dataset

Ship of Theseus Paraphrased Corpus:
- 7 datasets
- 4 paraphrasers: ChatGPT, PaLM2, Pegasus, and Dipper
- Iterative paraphrasing from T0 to T3

## Repository Structure

```
.
├── data/            # Ship of Theseus Paraphrased Corpus (T0–T3, 4 paraphrasers)
├── docs/            # Project documentation and notes
├── experiments/     # Experiment configs, logs, and results
├── figures/         # Plots and visualizations for analysis/paper
├── notebooks/       # Jupyter notebooks for exploration and analysis
├── paper/           # Manuscript and related writing
├── src/             # Source code (preprocessing, paraphrasing, analysis)
├── requirements.txt # Python dependencies
└── README.md
```

# Project Update 1

## Data
We use:
- Yelp
- XSum
- `source == "Human"`

Paraphrasers:
- ChatGPT
- PaLM
- Pegasus(full)
- Dipper

Only complete **T0 → T3** chains are retained.

## Processed Data

Main file:

```text
data/processed/update1_processed.csv
```

Columns:

```text
dataset, key, source, paraphraser, t0, t1, t2, t3
```

## Complete Chains

| Dataset | ChatGPT | PaLM | Pegasus | Dipper |
|---|---:|---:|---:|---:|
| Yelp | 488 | 486 | 491 | 491 |
| XSum | 471 | 340 | 475 | 476 |

## Code

```text
src/preprocessing.py
notebooks/01_data_exploration.ipynb
```

## Next Steps

- BLEU / ROUGE
- BERTScore
- Stylometric analysis
- Linguistic Delta

## Similarity Baseline (Member B)

The reproducible baseline compares each paraphrased generation with its original
`t0` text. It reports sentence-level BLEU, ROUGE-1/2/L F1, and BERTScore
precision/recall/F1 for `t1`, `t2`, and `t3`.

Install the dependencies and run:

```bash
python -m pip install -r requirements.txt
python src/similarity_baseline.py
```

For a faster first pass, run lexical metrics independently:

```bash
python src/similarity_baseline.py --metrics lexical \
  --details-output data/results/update1_lexical_scores.csv \
  --summary-output data/results/update1_lexical_summary.csv
```

Use `--limit 100` for a pilot run before processing the full dataset.

The default command reads `data/processed/update1_processed.csv` and writes:

- `data/results/update1_similarity_scores.csv`: one row per document, paraphraser, and generation.
- `data/results/update1_similarity_summary.csv`: mean, sample standard deviation, and count grouped by dataset, paraphraser, and generation.

The default BERTScore model is `distilbert-base-uncased` with `lang=en` and
CPU execution. On a CUDA-enabled machine, use `--device cuda` and increase
`--batch-size` as memory allows.
Scores are measured against `t0`, so higher values indicate greater lexical or
semantic retention relative to the human source.
