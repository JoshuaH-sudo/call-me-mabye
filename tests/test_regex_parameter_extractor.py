"""Unit tests for regex candidate extraction."""

from src.decoder.extractors.regex import RegexParameterExtractor


class TestRegexParameterExtractor:
    """Validates common regex scenarios and fallback behavior."""

    def setup_method(self) -> None:
        self.extractor = RegexParameterExtractor()

    def test_extracts_email_pattern(self) -> None:
        candidates = list(
            self.extractor.extract_candidates(
                "Provide a regex for common email address validation"
            )
        )
        assert (
            candidates[0]
            == r"^([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6})*$"
        )

    def test_extracts_phone_pattern_with_separators(self) -> None:
        candidates = list(
            self.extractor.extract_candidates(
                "Regex for US and International phone numbers with separators"
            )
        )
        assert (
            candidates[0]
            == r"^(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}$"
        )

    def test_extracts_zip_code_pattern(self) -> None:
        candidates = list(
            self.extractor.extract_candidates("Need a US postal code pattern")
        )
        assert candidates[0] == r"^\d{5}([\-]?\d{4})?$"

    def test_extracts_yyyy_mm_dd_date_pattern(self) -> None:
        candidates = list(
            self.extractor.extract_candidates(
                "Give me a regex for date format YYYY-MM-dd"
            )
        )
        assert (
            candidates[0]
            == (
                r"^(19|20)\d\d([- /.])(0[1-9]|1[012])"
                r"\2(0[1-9]|[12][0-9]|3[01])$"
            )
        )

    def test_extracts_hh_mm_ss_time_pattern(self) -> None:
        candidates = list(
            self.extractor.extract_candidates(
                "Need a regex for time format HH:MM:SS"
            )
        )
        assert (
            candidates[0]
            == r"^(?:[01]\d|2[0123]):(?:[012345]\d):(?:[012345]\d)$"
        )

    def test_extracts_ssn_without_dashes_when_requested(self) -> None:
        candidates = list(
            self.extractor.extract_candidates(
                "Regex for SSN without dashes"
            )
        )
        assert (
            candidates[0]
            == (
                r"^(?!219099999|078051120)(?!666|000|9\d{2})"
                r"\d{3}(?!00)\d{2}(?!0{4})\d{4}$"
            )
        )

    def test_keywords_still_work_for_vowels(self) -> None:
        candidates = list(
            self.extractor.extract_candidates(
                "Replace all vowels in Programming is fun"
            )
        )
        assert candidates[0] == "[aeiouAEIOU]"

    def test_number_keyword_prioritized_over_incidental_quotes(self) -> None:
        candidates = list(
            self.extractor.extract_candidates(
                (
                    'Replace all numbers in "Hello 34 I\'m 233 years old" '
                    "with NUMBERS"
                )
            )
        )
        assert candidates[0] == r"\d+"
