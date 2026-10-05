"""Checks for feature definitions, zero baselines, and exact score alignment."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.style_decay import FEATURES, KEYS, LEXICAL, MODELS, SEMANTIC, build_details, relative_values, stylistic_features


class StyleDecayTests(unittest.TestCase):
    def test_features_and_sentence_variance(self):
        values = stylistic_features('Cat cat. A dog runs!')
        self.assertAlmostEqual(values['type_token_ratio'], 4 / 5)
        self.assertEqual(values['sentence_length_mean'], 2.5)
        self.assertEqual(values['sentence_length_variance'], .25)
        self.assertEqual(values['punctuation_frequency'], 40)
        self.assertEqual(stylistic_features('One sentence.')['sentence_length_variance'], 0)

    def test_zero_baseline_and_direction(self):
        self.assertEqual(relative_values(0, 0), (0, 1))
        change, retention = relative_values(0, 2)
        self.assertTrue(np.isnan(change))
        self.assertEqual(retention, 0)
        self.assertEqual(relative_values(2, 4)[0], 1)
        self.assertEqual(relative_values(2, 1)[0], -.5)
        self.assertAlmostEqual(relative_values(2, 4)[1], relative_values(2, 1)[1])

    def test_alignment_anchors_and_missing_scores(self):
        chains = pd.DataFrame([dict(dataset='test', key='shared', source='Human', paraphraser=m,
                                    t0='A cat.', t1='A dog.', t2='A bird.', t3='A horse.') for m in MODELS])
        with tempfile.TemporaryDirectory() as directory:
            rows = [{**{k: c[k] for k in KEYS[:-1]}, 'generation': g} for c in chains.to_dict('records') for g in ['t1', 't2', 't3']]
            lexical = pd.DataFrame(rows).assign(**{m: .5 for m in LEXICAL})
            semantic = pd.DataFrame(rows).assign(**{m: .8 for m in SEMANTIC})
            lp, sp = Path(directory) / 'lex.csv', Path(directory) / 'sem.csv'
            lexical.iloc[::-1].to_csv(lp, index=False)
            semantic.to_csv(sp, index=False)
            result = build_details(chains, lp, sp)
            self.assertEqual(len(result), 16)
            self.assertTrue(result.matched_cohort.all())
            self.assertTrue(result.loc[result.generation.eq('t0'), LEXICAL + SEMANTIC].eq(1).all().all())
            self.assertTrue(result.loc[result.generation.eq('t1'), 'bleu'].eq(.5).all())
            self.assertTrue(result[[f + '_retention' for f in FEATURES]].eq(1).all().all())
            semantic.iloc[:-1].to_csv(sp, index=False)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                build_details(chains, lp, sp)

    def test_duplicate_chains_rejected(self):
        chains = pd.DataFrame([dict(dataset='test', key='shared', source='Human', paraphraser=m,
                                    t0='A cat.', t1='A dog.', t2='A bird.', t3='A horse.') for m in MODELS])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            build_details(pd.concat([chains, chains]), Path('unused'), Path('unused'))


if __name__ == '__main__':
    unittest.main()
