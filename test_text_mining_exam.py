from pathlib import Path
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.utils.testing import ignore_warnings

from text_mining_exam import split_paragraphs
from text_mining_exam import parse_filename
from text_mining_exam import load_text_file
from text_mining_exam import prepare_dataset
from text_mining_exam import make_language_classifier


HERE = Path(__file__).parent
DATA_FOLDER = HERE / 'data'

TEST_TEXT = """quit()

Header

This is the content of the first paragraph. This paragrah is long enough
to span two lines.

The second paragraph is much shorter.

"""


def test_split_paragraphs():
    paragraphs = split_paragraphs(TEST_TEXT)
    assert len(paragraphs) == 2
    assert paragraphs[0].startswith("This is the content of the first")
    assert paragraphs[0].endswith("two lines.")
    assert paragraphs[1] == "The second paragraph is much shorter."


def test_parse_filename():
    filepath = DATA_FOLDER / "english__utf8.txt"
    language, encoding = parse_filename(filepath)
    assert language == "english"
    assert encoding == "utf8"

    filepath = DATA_FOLDER / "french__iso_8859_15.txt"
    language, encoding = parse_filename(filepath)
    assert language == "french"
    assert encoding == "iso_8859_15"


def test_load_english_text_file():
    text = load_text_file(DATA_FOLDER / "english__utf8.txt")
    assert text.startswith(
        "Science (from Latin scientia, meaning \"knowledge\")")


def test_load_text_file_french():
    text = load_text_file(DATA_FOLDER / "french__iso_8859_15.txt")
    assert text.startswith(
        "La science est l'ensemble des connaissances et études "
        "d'une valeur universelle")
    assert "François Rabelais" in text


def test_prepare_dataset():
    paragraphs, languages = prepare_dataset(DATA_FOLDER.glob("*.txt"))
    assert len(paragraphs) == len(languages)

    label_counts = Counter(languages)
    assert len(label_counts) == 2
    assert 100 <= label_counts["french"] <= 300
    assert 50 <= label_counts["english"] <= 100

    for paragraphs in paragraphs:
        assert 5 <= len(paragraphs) < 10000


@ignore_warnings(category=PendingDeprecationWarning)
def test_language_classifier():
    paragraphs, languages = prepare_dataset(DATA_FOLDER.glob("*.txt"))
    clf = make_language_classifier()
    assert cross_val_score(clf, paragraphs, languages, cv=5).mean() >= 0.9

    assert clf.fit(paragraphs, languages)
    assert clf.predict(["C'est vraiment l'été!"]).tolist() == ["french"]
    assert clf.predict(["This is really working!"]).tolist() == ["english"]
