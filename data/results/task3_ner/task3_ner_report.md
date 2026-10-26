# Task 3 — NER Stability & Content Erasure

## Completion and verification

The analysis covers 18,049 article-source pairs across seven datasets and seven source groups, producing 72,196 comparison rows for four T1 variants. The supplied verification summary records that all four metric edge-case tests passed, every exported score and all main summary statistics were independently recomputed.

## Method and exclusions

Compare each T1 variant against its aligned T0 using all available source groups. Extract entity surface strings across all spaCy entity labels using en_core_web_sm. Lowercase and strip each string, then deduplicate; repeated mentions count once.

- Jaccard = intersection / union.
- Recall = intersection / T0 entity-set size.
- Precision = intersection / T1 entity-set size.

When T0 has zero entities, all three metrics are NaN and the article-source pair is excluded from every metric mean. With nonempty T0 and empty T1, Jaccard and recall are 0, while precision is undefined (NaN) and excluded separately. Valid counts are reported per metric. Standard deviation uses ddof=1 and is undefined when n < 2.

Of the 18,049 original article-source texts, 1,600 have no recognized entities (8.86%). Excluding them leaves 16,449 eligible pairs per variant for Jaccard and recall. Precision additionally excludes empty rewrites. The nonempty-original summary uses the same eligible cohort as the main analysis and matches the main dataset summary.

Overall results weight article-source pairs equally. Different source pairs may share article keys; no independence or statistical significance is claimed.

Supporting files: zero_entity_counts.csv and nonempty_original_ner_summary.csv.

## Comparison tables (mean ± sample SD)

| variant | jaccard (n) | recall (n) | precision (n) |
| --- | --- | --- | --- |
| ChatGPT | 0.551 ± 0.301 (16449) | 0.629 ± 0.306 (16449) | 0.764 ± 0.256 (15797) |
| Dipper(high) | 0.021 ± 0.063 (16449) | 0.037 ± 0.112 (16449) | 0.058 ± 0.144 (14326) |
| Pegasus(full) | 0.517 ± 0.287 (16449) | 0.558 ± 0.292 (16449) | 0.845 ± 0.220 (15284) |
| Pegasus(slight) | 0.868 ± 0.203 (16449) | 0.887 ± 0.192 (16449) | 0.967 ± 0.099 (16219) |

| dataset | variant | jaccard (n) | recall (n) | precision (n) |
| --- | --- | --- | --- | --- |
| cmv | ChatGPT | 0.488 ± 0.312 (1365) | 0.560 ± 0.322 (1365) | 0.717 ± 0.295 (1292) |
| cmv | Dipper(high) | 0.028 ± 0.069 (1365) | 0.054 ± 0.132 (1365) | 0.069 ± 0.150 (1235) |
| cmv | Pegasus(full) | 0.530 ± 0.291 (1365) | 0.577 ± 0.296 (1365) | 0.829 ± 0.234 (1266) |
| cmv | Pegasus(slight) | 0.876 ± 0.194 (1365) | 0.899 ± 0.184 (1365) | 0.965 ± 0.100 (1343) |
| eli5 | ChatGPT | 0.464 ± 0.333 (2499) | 0.529 ± 0.347 (2499) | 0.718 ± 0.313 (2264) |
| eli5 | Dipper(high) | 0.026 ± 0.078 (2499) | 0.049 ± 0.138 (2499) | 0.064 ± 0.151 (2145) |
| eli5 | Pegasus(full) | 0.512 ± 0.322 (2499) | 0.552 ± 0.329 (2499) | 0.837 ± 0.250 (2212) |
| eli5 | Pegasus(slight) | 0.864 ± 0.228 (2499) | 0.884 ± 0.217 (2499) | 0.965 ± 0.109 (2431) |
| sci_gen | ChatGPT | 0.584 ± 0.353 (2453) | 0.635 ± 0.358 (2453) | 0.840 ± 0.237 (2357) |
| sci_gen | Dipper(high) | 0.018 ± 0.051 (2453) | 0.032 ± 0.104 (2453) | 0.055 ± 0.136 (2181) |
| sci_gen | Pegasus(full) | 0.472 ± 0.318 (2453) | 0.503 ± 0.325 (2453) | 0.853 ± 0.222 (2218) |
| sci_gen | Pegasus(slight) | 0.879 ± 0.196 (2453) | 0.896 ± 0.187 (2453) | 0.975 ± 0.083 (2420) |
| tldr | ChatGPT | 0.626 ± 0.254 (2312) | 0.713 ± 0.244 (2312) | 0.806 ± 0.205 (2290) |
| tldr | Dipper(high) | 0.008 ± 0.034 (2312) | 0.013 ± 0.061 (2312) | 0.029 ± 0.101 (1933) |
| tldr | Pegasus(full) | 0.512 ± 0.262 (2312) | 0.550 ± 0.267 (2312) | 0.850 ± 0.215 (2187) |
| tldr | Pegasus(slight) | 0.865 ± 0.194 (2312) | 0.882 ± 0.184 (2312) | 0.969 ± 0.095 (2289) |
| wp | ChatGPT | 0.515 ± 0.294 (1851) | 0.599 ± 0.302 (1851) | 0.732 ± 0.267 (1758) |
| wp | Dipper(high) | 0.048 ± 0.092 (1851) | 0.085 ± 0.164 (1851) | 0.107 ± 0.189 (1681) |
| wp | Pegasus(full) | 0.534 ± 0.284 (1851) | 0.584 ± 0.291 (1851) | 0.837 ± 0.218 (1711) |
| wp | Pegasus(slight) | 0.884 ± 0.183 (1851) | 0.907 ± 0.168 (1851) | 0.967 ± 0.095 (1835) |
| xsum | ChatGPT | 0.616 ± 0.210 (3025) | 0.707 ± 0.201 (3025) | 0.802 ± 0.164 (3017) |
| xsum | Dipper(high) | 0.010 ± 0.028 (3025) | 0.013 ± 0.043 (3025) | 0.050 ± 0.131 (2600) |
| xsum | Pegasus(full) | 0.525 ± 0.205 (3025) | 0.563 ± 0.203 (3025) | 0.858 ± 0.168 (2990) |
| xsum | Pegasus(slight) | 0.836 ± 0.198 (3025) | 0.855 ± 0.187 (3025) | 0.958 ± 0.104 (3010) |
| yelp | ChatGPT | 0.522 ± 0.307 (2944) | 0.615 ± 0.310 (2944) | 0.703 ± 0.285 (2819) |
| yelp | Dipper(high) | 0.019 ± 0.071 (2944) | 0.034 ± 0.110 (2944) | 0.048 ± 0.139 (2551) |
| yelp | Pegasus(full) | 0.539 ± 0.312 (2944) | 0.582 ± 0.316 (2944) | 0.837 ± 0.238 (2700) |
| yelp | Pegasus(slight) | 0.883 ± 0.209 (2944) | 0.900 ± 0.199 (2944) | 0.970 ± 0.104 (2891) |

## Pegasus controlled comparison

Pegasus(slight) rewrites a random 25% of sentences; Pegasus(full) rewrites all sentences. The model and random seed are held fixed and sentence coverage differs. Upstream generation settings were not independently verified.

The chart pools eligible article-source pairs across all seven datasets. It has three metric clusters (Jaccard, Recall, Precision), two bars per cluster (slight and full), labeled axes, and article-level sample SD error bars. Per-dataset comparisons remain available in the CSV tables.

Chart file: pegasus_ner_comparison.png.

Differences are paired by dataset, key and source. See pegasus_full_minus_slight_summary.csv for full-minus-slight mean differences and sample SD. Negative values indicate lower overlap under full paraphrasing. Each metric excludes pairs where either value is undefined.

## Interpretation

Among 16,449 eligible article-source pairs, mean Jaccard was 0.551 for ChatGPT, 0.021 for Dipper(high), 0.868 for Pegasus(slight), and 0.517 for Pegasus(full). The Pegasus slight-to-full decline was larger for Recall (0.329) than Precision (0.122), indicating more entity dropping than introduction of new entity strings. This relationship held across 7 of 7 datasets, although its magnitude varied by domain: Recall gaps were 0.392 for scientific text, 0.292 for news, and 0.318 for reviews. Thus, greater paraphrase coverage primarily affected entity retention. Exact-string differences can reflect aliases or NER errors rather than factual hallucinations. Dipper(high)'s low overlap warrants checking generation quality and upstream alignment.
