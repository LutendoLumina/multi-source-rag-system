import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingest import normalize_extracted_text, strip_page_number_artifact


def test_strip_page_number_artifact_removes_leading_digits():
    assert strip_page_number_artifact("16We understand the importance") == "We understand the importance"


def test_strip_page_number_artifact_leaves_real_text_alone():
    text = "We understand 16 things about finance"
    assert strip_page_number_artifact(text) == text


def test_normalize_collapses_character_spaced_runs():
    assert normalize_extracted_text("T h i s   i s   a   t e s t") == "This is a test"


def test_normalize_preserves_short_real_words():
    text = "I am a student"
    assert normalize_extracted_text(text) == text


def test_normalize_fixes_punctuation_spacing():
    assert normalize_extracted_text("35 % complete , great") == "35% complete, great"


def test_normalize_strips_page_number_and_collapses_run():
    raw = "13F i n a l   P r o j e c t counts for 35 % of the grade"
    assert normalize_extracted_text(raw) == "Final Project counts for 35% of the grade"


def test_normalize_collapses_extra_whitespace():
    assert normalize_extracted_text("Too    many     spaces\nand a newline") == "Too many spaces and a newline"
