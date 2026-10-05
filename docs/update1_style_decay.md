# Project Update 1 — Style vs. Content Decay

This analysis extends the existing similarity baseline with initial stylistic features. We compare complete Human T0 → T3 chains from Yelp and XSum for ChatGPT, PaLM, Dipper, and Pegasus(full). PaLM is displayed as PaLM2 (PaLM) in the figures to connect the project terminology with the stored corpus label; this analysis does not independently verify the underlying model version.

## 1. Analysis Scope

| dataset | paraphraser | complete_chains |
| --- | --- | --- |
| xsum | ChatGPT | 471 |
| xsum | Dipper | 476 |
| xsum | PaLM | 340 |
| xsum | Pegasus | 475 |
| yelp | ChatGPT | 488 |
| yelp | Dipper | 491 |
| yelp | PaLM | 486 |
| yelp | Pegasus | 491 |

All complete chains are retained in the main outputs. Because chain availability differs across paraphrasers, comparisons are also repeated on source documents available for all four paraphrasers: XSUM = 336, YELP = 483. Each matched source has the same T0 across models. Datasets are analyzed separately.

## 2. Initial Stylistic Features

- Type–Token Ratio: unique lowercase word tokens divided by all word tokens. Apostrophes within words are retained; numbers are excluded.
- Mean sentence length: mean word-token count across nonempty sentences.
- Sentence length variance: population variance of sentence word-token counts (ddof = 0); a single sentence has variance 0.
- Punctuation frequency: Unicode punctuation characters per 100 word tokens.

Sentence boundaries use a lightweight rule based on sentence-final punctuation followed by whitespace, or line breaks. Abbreviations and irregular punctuation can affect segmentation. TTR is sensitive to text length. These features provide initial descriptive evidence, not a validated measure of authorial identity.

## 3. T0-relative Metrics

BLEU and ROUGE-1/2/L F1 represent lexical retention. BERTScore precision/recall/F1 represent semantic similarity, with F1 used in the main figures. The supplied document-level score files are reused after exact key and coverage validation. The existing baseline defaults to distilbert-base-uncased without baseline rescaling; the supplied CSV files do not record the actual model, configuration, or package versions used, so their generation settings cannot be independently confirmed.

For similarity metrics, T0 self-similarity is set to its theoretical value of 1. Retention is S(Tg,T0) / S(T0,T0), which equals the stored score; relative change is retention − 1. T0 anchors are not newly computed BERTScore observations.

For each stylistic feature f, signed relative change is (f(Tg) − f(T0)) / f(T0). If both values are zero, change is 0; if only T0 is zero, the ratio is undefined and saved as a missing value. Per-metric counts in the summary expose these exclusions.

To include zero-baseline features in a bounded descriptive curve, feature retention is 1 − |f(Tg) − f(T0)| / (|f(Tg)| + |f(T0)|), with retention 1 when both are zero. Style retention is the unweighted mean of the four feature retentions for each chain and generation. This is an exploratory feature-preservation index. Normalization gives a common T0 anchor, but does not make BLEU, BERTScore, and style retention empirically calibrated or directly equivalent.

## 4. Decay Curves

![Lexical retention](../figures/update1_lexical_decay.png)

**Figure 1.** Mean BLEU and ROUGE-1/2/L F1 relative to the original T0 text. Each row is a dataset and each line is a paraphraser. Lower values indicate less surface overlap; the same complete chains contribute at every generation within each line.

![Stylistic feature retention](../figures/update1_style_decay.png)

**Figure 2.** T0-relative retention of each initial stylistic feature. A lower value means the feature has moved farther from its source value in either direction. Recovery is possible, so these trajectories should not be assumed to decay monotonically.

![Retention comparison](../figures/update1_retention_comparison.png)

**Figure 3.** BLEU, ROUGE-L F1, BERTScore F1, and exploratory style retention for all complete chains. Vertical differences across metric families are descriptive because their scales are not calibrated.

![Matched comparison](../figures/update1_matched_comparison.png)

**Figure 4.** The same comparison restricted to source documents with complete chains for all four paraphrasers. This controls source composition within each dataset. Lines show means; uncertainty is not represented by confidence bands.

## 5. Results

T3 means on the matched cohort:

| dataset | paraphraser | bleu | rougeL_f1 | bertscore_f1 | style_retention |
| --- | --- | --- | --- | --- | --- |
| xsum | ChatGPT | 0.201 | 0.447 | 0.874 | 0.864 |
| xsum | Dipper | 0.084 | 0.245 | 0.823 | 0.859 |
| xsum | PaLM | 0.435 | 0.652 | 0.919 | 0.889 |
| xsum | Pegasus | 0.248 | 0.516 | 0.852 | 0.825 |
| yelp | ChatGPT | 0.127 | 0.390 | 0.845 | 0.831 |
| yelp | Dipper | 0.073 | 0.250 | 0.814 | 0.828 |
| yelp | PaLM | 0.203 | 0.480 | 0.864 | 0.830 |
| yelp | Pegasus | 0.221 | 0.505 | 0.865 | 0.801 |

### XSUM

ChatGPT: from T1 to T3, BLEU changes from 0.262 to 0.201, ROUGE-L F1 from 0.523 to 0.447, BERTScore F1 from 0.892 to 0.874, and exploratory style retention from 0.875 to 0.864.

PaLM: from T1 to T3, BLEU changes from 0.545 to 0.435, ROUGE-L F1 from 0.738 to 0.652, BERTScore F1 from 0.937 to 0.919, and exploratory style retention from 0.911 to 0.889.

Dipper: from T1 to T3, BLEU changes from 0.208 to 0.084, ROUGE-L F1 from 0.357 to 0.245, BERTScore F1 from 0.866 to 0.823, and exploratory style retention from 0.879 to 0.859.

Pegasus: from T1 to T3, BLEU changes from 0.410 to 0.248, ROUGE-L F1 from 0.653 to 0.516, BERTScore F1 from 0.897 to 0.852, and exploratory style retention from 0.867 to 0.825.

At T3, BERTScore F1 ranks PaLM > ChatGPT > Pegasus > Dipper on the matched sources. This describes this metric and sample, rather than overall paraphraser quality.

### YELP

ChatGPT: from T1 to T3, BLEU changes from 0.179 to 0.127, ROUGE-L F1 from 0.475 to 0.390, BERTScore F1 from 0.869 to 0.845, and exploratory style retention from 0.850 to 0.831.

PaLM: from T1 to T3, BLEU changes from 0.293 to 0.203, ROUGE-L F1 from 0.595 to 0.480, BERTScore F1 from 0.890 to 0.864, and exploratory style retention from 0.847 to 0.830.

Dipper: from T1 to T3, BLEU changes from 0.172 to 0.073, ROUGE-L F1 from 0.365 to 0.250, BERTScore F1 from 0.857 to 0.814, and exploratory style retention from 0.863 to 0.828.

Pegasus: from T1 to T3, BLEU changes from 0.370 to 0.221, ROUGE-L F1 from 0.647 to 0.505, BERTScore F1 from 0.905 to 0.865, and exploratory style retention from 0.862 to 0.801.

At T3, BERTScore F1 ranks Pegasus > PaLM > ChatGPT > Dipper on the matched sources. This describes this metric and sample, rather than overall paraphraser quality.

### Initial Stylistic Patterns

XSUM: across the four paraphrasers at T3, TTR retention ranges from 0.941 to 0.970, mean sentence length retention from 0.782 to 0.929, sentence length variance retention from 0.681 to 0.756, and punctuation retention from 0.833 to 0.924. These values use all complete chains, as in Figure 2.

YELP: across the four paraphrasers at T3, TTR retention ranges from 0.960 to 0.966, mean sentence length retention from 0.814 to 0.877, sentence length variance retention from 0.601 to 0.663, and punctuation retention from 0.816 to 0.844. These values use all complete chains, as in Figure 2.

Sentence length variance shows the largest departure under this feature-distance definition, while TTR stays closest to its source value. Variance is sensitive to sentence segmentation and zero baselines, so this difference requires follow-up before being interpreted as a robust stylistic effect.



