# -*- coding: utf-8 -*-
"""Tests du support des labels abjad (arabe) via abjad.yaml.

Les marqueurs arabes (أ، ب، ج…) sont translitteres en latin (a, b, c…) pour
reutiliser la detection existante, puis restitues en forme abjadiya a
l'ecriture du JSON (latin_to_abjad).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from label_sequence import (  # noqa: E402
    _canonical_marker,
    _classify_marker,
    _extract_marker_questions,
    _extract_markers,
    build_marker_pairs,
    classify_marker_pairs,
    latin_to_abjad,
)


class TestAbjadExtraction(unittest.TestCase):
    def test_extract_markers(self):
        prompt = "**أ)** السؤال الأول<br>**ب)** السؤال الثاني<br>**1.أ)** سؤال فرعي"
        self.assertEqual(_extract_markers(prompt), ["a)", "b)", "1.a)"])

    def test_markers_du_ligne_sans_gras(self):
        prompt = "أ) السؤال الأول\nب) السؤال الثاني"
        self.assertEqual(_extract_markers(prompt), ["a)", "b)"])

    def test_questions_clefs_latines(self):
        prompt = "**أ)** السؤال الأول<br>**ب)** السؤال الثاني"
        qs = _extract_marker_questions(prompt)
        self.assertEqual(set(qs.keys()), {"a)", "b)"})
        self.assertIn("السؤال الأول", qs["a)"])

    def test_texte_arabe_non_marqueur_inchange(self):
        text = "الماء نعمة من نعم الله"
        self.assertEqual(_extract_markers(text), [])
        self.assertEqual(_abjad_to_latin_guard(text), text)


class TestAbjadClassification(unittest.TestCase):
    def test_lettres(self):
        key = _classify_marker("أ)")
        self.assertIsNotNone(key)
        self.assertEqual(key.kind, "letter")
        self.assertEqual(key.value, (0,))
        self.assertEqual(_canonical_marker(key), "a")
        self.assertEqual(_classify_marker("ب)").value, (1,))

    def test_lettres_sous_partie(self):
        key = _classify_marker("1.ج)")
        self.assertIsNotNone(key)
        self.assertEqual(key.sub, 2)
        self.assertEqual(_canonical_marker(key), "1.c")

    def test_chiffres_arabes_indic(self):
        key = _classify_marker("١)")
        self.assertIsNotNone(key)
        self.assertEqual(key.value, (1,))
        self.assertEqual(_canonical_marker(key), "1")


class TestAbjadMatching(unittest.TestCase):
    def test_matched(self):
        groups = classify_marker_pairs(
            ["أ)", "ب)", "1.ج)"], ["أ)", "ب)", "1.ج)"]
        )
        self.assertTrue(all(g.kind == "MATCHED" for g in groups))

    def test_missing(self):
        prompt = "**أ)** سؤال<br>**ب)** سؤال"
        markers = _extract_markers(prompt)
        self.assertEqual(markers, ["a)", "b)"])
        groups = classify_marker_pairs(markers, ["أ)"])
        self.assertEqual(groups[0].kind, "MATCHED")
        self.assertEqual(groups[1].kind, "MISSING")
        self.assertEqual(groups[1].missing_markers, ["b)"])

    def test_build_marker_pairs(self):
        prompt = "**أ)** سؤال<br>**ب)** سؤال<br>**1.ج)** سؤال"
        pairs = build_marker_pairs(prompt, ["أ)", "ب)", "1.ج)"], ["r1", "r2", "r3"])
        markers = [p["marqueur"] for p in pairs["marqueurs_questions"]]
        self.assertEqual(markers, ["a)", "b)", "1.c)"])


class TestAbjadRestore(unittest.TestCase):
    def test_latin_to_abjad(self):
        self.assertEqual(latin_to_abjad("a)"), "أ)")
        self.assertEqual(latin_to_abjad("1.a)"), "1.أ)")
        self.assertEqual(latin_to_abjad("I.1.a)"), "I.1.أ)")
        self.assertEqual(latin_to_abjad("I.2)"), "I.2)")

    def test_normalize_puis_restaure(self):
        from correction import _normalize_dcc_marker  # noqa: PLC0415

        self.assertEqual(_normalize_dcc_marker("أ)"), "a)")
        self.assertEqual(latin_to_abjad(_normalize_dcc_marker("أ)")), "أ)")
        self.assertEqual(latin_to_abjad(_normalize_dcc_marker("1.ج)")), "1.ج)")


def _abjad_to_latin_guard(text):
    from label_sequence import _abjad_to_latin_markers  # noqa: PLC0415

    return _abjad_to_latin_markers(text)


if __name__ == "__main__":
    unittest.main()
