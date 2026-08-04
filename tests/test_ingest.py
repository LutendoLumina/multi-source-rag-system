"""
Unit tests for ingest.py — Part 1 (Process the Handbook).

Covers the text-cleaning helpers that fix PDF-extraction artifacts before
chunking/embedding. No PDF, embedding model, or vector DB is needed to run
these — they test pure text-transformation functions in isolation.

Test fixtures mirror this handbook's actual PyPDFLoader extraction
convention (verified against the real handbook.pdf): individual characters
within a word are separated by a single space (e.g. "T h i s"), real word
boundaries use a double space, and the page-number footer is separated
from the first word of content by a newline (e.g. "1 6\nW e  u n d e r").
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingest import normalize_extracted_text, strip_page_number_artifact


class TestStripPageNumberArtifact:

    def test_removes_leading_number_separated_by_newline(self):
        assert strip_page_number_artifact("16\nWe understand the importance") == "We understand the importance"

    def test_removes_leading_number_separated_by_space(self):
        assert strip_page_number_artifact("4 Hardware requirements") == "Hardware requirements"

    def test_removes_short_and_long_leading_numbers(self):
        assert strip_page_number_artifact("4\nHardware requirements") == "Hardware requirements"
        assert strip_page_number_artifact("123\nSome content") == "Some content"

    def test_leaves_real_text_with_embedded_numbers_alone(self):
        text = "We understand 16 things about finance"
        assert strip_page_number_artifact(text) == text

    def test_leaves_text_with_no_leading_number_alone(self):
        text = "Final Project counts for 35% of the grade"
        assert strip_page_number_artifact(text) == text

    def test_does_not_strip_a_number_glued_directly_to_letters(self):
        text = "256GB of storage recommended"
        assert strip_page_number_artifact(text) == text


class TestNormalizeExtractedText:

    def test_collapses_character_spaced_words(self):
        assert normalize_extracted_text("T h i s   i s   a   t e s t") == "This is a test"

    def test_preserves_short_real_words(self):
        text = "I am a student"
        assert normalize_extracted_text(text) == text

    def test_fixes_punctuation_spacing(self):
        assert normalize_extracted_text("35 % complete , great") == "35% complete, great"

    def test_does_not_bridge_across_a_real_paragraph_break(self):
        raw = "1 6\nW e  u n d e r s t a n d"
        assert normalize_extracted_text(raw) == "We understand"

    def test_strips_real_handbook_page_number_artifact(self):
        raw = "1 6\nW e  u n d e r s t a n d  t h e  i m p o r t a n c e"
        assert normalize_extracted_text(raw) == "We understand the importance"

    def test_collapses_extra_whitespace_and_newlines(self):
        assert normalize_extracted_text("Too    many     spaces\nand a newline") == "Too many spaces and a newline"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_extracted_text("   padded text   ") == "padded text"

    def test_matches_real_handbook_fees_page_output(self):
        raw = ("1 6\nW e  u n d e r s t a n d  t h e  i m p o r t a n c e  o f  "
               "f i n a n c i a l  p l a n n i n g ,  a n d  w e  a i m  t o  "
               "m a k e")
        result = normalize_extracted_text(raw)
        assert "aim tomake" not in result
        assert "aim to make" in result
        assert result.startswith("We understand")