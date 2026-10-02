import unittest

from humanizer_server import (
    ai_risk_score,
    analyze,
    count_words,
    detect_language,
    missing_keywords,
    validate_candidate,
)


class HumanizerAnalyticsTests(unittest.TestCase):
    def test_counts_words_and_detects_languages(self):
        self.assertEqual(count_words("Natural writing has rhythm."), 4)
        self.assertEqual(detect_language("This is an English sentence."), "en")
        self.assertEqual(detect_language("هذه جملة عربية."), "ar")

    def test_analysis_identifies_ai_cliches(self):
        metrics = analyze(
            "We leverage a comprehensive solution. "
            "This robust approach delivers results.",
            "en",
        )
        self.assertGreater(metrics.trigger_hits, 0)
        self.assertGreater(metrics.risk_score, 0)

    def test_keyword_matching_is_case_and_diacritic_insensitive(self):
        self.assertEqual(
            missing_keywords("Machine Learning improves results.", ["machine learning"]),
            [],
        )
        self.assertEqual(missing_keywords("A short text.", ["Redis"]), ["Redis"])

    def test_candidate_validation_enforces_range_and_keywords(self):
        valid, word_count, missing, penalty = validate_candidate(
            "Keep Redis available for caching.", 5, 5, ["Redis"]
        )
        self.assertTrue(valid)
        self.assertEqual(word_count, 5)
        self.assertEqual(missing, [])
        self.assertEqual(penalty, 0)

        valid, _, missing, penalty = validate_candidate(
            "Keep caching available.", 3, 5, ["Redis"]
        )
        self.assertFalse(valid)
        self.assertEqual(missing, ["Redis"])
        self.assertGreater(penalty, 0)

    def test_risk_score_is_bounded(self):
        self.assertEqual(ai_risk_score(0), 0.0)
        self.assertGreaterEqual(ai_risk_score(100), 0)
        self.assertLessEqual(ai_risk_score(100), 100)


if __name__ == "__main__":
    unittest.main()
