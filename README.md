# Ship of Theseus: Computational Forensics

This repository studies how repeated paraphrasing changes writing style, semantic content, and source-identifying features from the original text (T0) to three successive rewrites (T1–T3). It includes the initial baselines from Project Update 1 and the expanded RQ1–RQ3 experiments for Project Update 2.

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

At what iteration does a text lose its original authorial identity? We train a
Human-vs.-LLM classifier on T0 and test it on held-out T0–T3 data, then compare
how much performance drops at each paraphrasing step.

### RQ3 — Paraphraser Fingerprints

Do ChatGPT, PaLM, Pegasus, and Dipper leave distinguishable linguistic traces?
This will be studied later with paraphraser classification and feature analysis.

## Project Overview

We use the [Ship of Theseus Paraphrased Corpus](https://github.com/tripto03/Ship_of_theseus_paraphrased_copus) across **7 datasets**, with Human and LLM original sources and **7 paraphraser configurations**.

- **132,659** complete T0–T3 chains; **530,636** text-version records.
- **23 linguistic features:** 17 POS proportions, dependency depth, and 5 stylometric features.
- **RQ1:** Lexical/semantic retention, POS-frequency and POS-distribution changes, dependency-depth drift.
- **RQ2:** Human-vs.-LLM source attribution across T0–T3 and PCA/t-SNE identity trajectories.
- **RQ3:** Initial T1 paraphraser classification (ChatGPT, PaLM, Dipper, Pegasus).

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Download [`update2_chains.pkl` and `update2_features.pkl` from Google Drive](https://drive.google.com/drive/folders/1ju91YAi1wfuECSzvz9RfPeho32v66HiC?usp=sharing), then place both in `data/processed/`. Raw and large processed datasets are not stored in GitHub.

## Run the Notebooks

Open the `notebooks/` folder in VS Code or Jupyter, select the project's Python environment as the kernel, and run in order:

| Notebook | Purpose |
|---|---|
| `04_update2_data_preparation.ipynb` | Validate T0–T3 chains and extracted features |
| `05_update2_rq1_style_analysis.ipynb` | RQ1: POS frequencies/drift, dependency depth, Human vs. LLM |
| `06_update2_rq2_authorship.ipynb` | RQ2: Train on T0; evaluate attribution on T0–T3 |
| `07_update2_rq2_identity_trajectory.ipynb` | RQ2: PCA and t-SNE visualizations |
| `08_update2_rq3_fingerprints.ipynb` | RQ3: Classifiers, confusion matrices, and feature analysis |

Notebooks `01`–`03` contain earlier baseline and linguistic-audit work; they are not prerequisites when using the downloaded Update 2 files.

Alternatively, run the RQ2–RQ3 modules from the repository root:

```bash
python -m src.authorship_attribution
python -m src.identity_trajectory
python -m src.paraphraser_fingerprint
```

## Outputs

- `data/results/update2/` — RQ1–RQ3 CSV/JSON results.
- `figures/update2/` — POS, dependency-depth, attribution, PCA/t-SNE, and classifier figures.
- `docs/update2_rq3_fingerprint.md` — RQ3 method and findings.
- `data/results/update1_*.csv` and `figures/update1_*.png` — original lexical/semantic baselines.

Notebook 05 also exports `pos_tag_frequency_by_stage.csv` and `pos_tag_frequency_shift.png` when its new POS-frequency section is run. These outputs are being added with the Member 1 updates.

## Rebuild Data (Optional)

To regenerate the shared Update 2 pickle files, place all seven raw CSVs in `data/raw/paraphrased_datasets/` (named `<dataset>_paraphrased.csv`) and run:

```bash
python -m src.chain_alignment
python -m src.stylometry
```

Full feature extraction may be time-consuming. To run automated checks:

```bash
python -m pytest -q
```
