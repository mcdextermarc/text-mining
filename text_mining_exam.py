from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def split_paragraphs(text, min_length=20):
    r"""Split the provided text into a list of paragraphs.

    Paragraphs are separated by double new line characters ("\n\n").

    Empty paragraphs and any paragraphs that are shorter than 20 symbols are
    skipped.
    """
    # Hints:
    # - use the `text.split(separator)` to obtain the list of sub-texts
    #   separated by some text `separator`;
    # - use `paragraph.strip()` to remove leading and trailing white-spaces
    #  (including new line characters).
    text = text.split("\n\n")
    paragraphs = [p for p in text if len(p) > min_length]
    return paragraphs


def parse_filename(filepath):
    """Retrieve the language and encoding from the filename

    Assume that the text file names follow a naming convention such as:

    language__encoding.txt

    For instance: "italian__utf8.txt" where the double "__" underscore
    separator separates the "italian" language from the "utf8" encoding.

    Return a tuple `(language, encoding)`.
    """
    # Hints:
    # - `filepath.name` gives the file name of a pathlib.Path object.
    # - use `filename.split(separator)` to return the list of filename elements
    #   using separator
    filepath_split = filepath.name
    filepath_split = filepath_split.split('__')
    language = filepath_split[0] 
    encoding = filepath_split[1][:-4]
    return (language, encoding)


def load_text_file(filepath):
    """Read the text content of a file using the encoding in its file name"""
    # Hints:
    # - Use `parse_filename` to find the encoding of the file.
    # - Use the `read_text` method of the `filepath` object.
    language_encoding = parse_filename(filepath)[1]
    return filepath.read_text(encoding=language_encoding)


def prepare_dataset(filepaths):
    """Prepare a dataset suitable to train a language classification model

    Assume that the file names follow the above naming convention:
    - read all the text files with the correct encoding,
    - split each text document to extract the paragraphs,
    - collect all the paragraphs and the matching language names for each
      paragraph.

    All the paragraphs of a given document are written in the language
    retrieved from the text file.

    Return the tuple (paragraphs, languages).
    """
    # Hints:
    # - Feel free to reuse previous functions.
    # - The returned list of paragraphs and languages should have the same
    #   length.
    
    paragraphs = []
    languages = []
    for path in filepaths:
        for text in split_paragraphs(load_text_file(path)):
                paragraphs.append(text)
                languages.append(parse_filename(path)[0])
    return paragraphs, languages


def make_language_classifier():
    """Create a (untrained) text classification pipeline

    The pipeline has 2 steps:

    - A `TfidfVectorizer` uses a character-based analyzer to extract n-grams
      of characters between with "n" ranging from 1 to 3.

      Character n-grams that appear less than 3 times in the dataset are
      ignored.

      Character n-grams that appear in more than 90% of the texts are also
      ignored.

      tf-idf vectors are normalized with the l2 norm to make the feature
      vectors less sensitive to the length of the text document.

    - A `LogisticRegression` classifier that classifies the extracted tf-idf
      features using the target class labels.

      The classifier uses the "lbfgs" solver.

    Return the pipeline object.
    """
    # Hints:
    # - read the documentation of TfidfVectorizer (online on scikit-learn.org
    #   or by reading the docstring in the source code of scikit-learn) to
    #   learn how to configure the character-based analyzer  and the n-gram
    #   extraction.
    # - Use make_pipeline to assemble the vectorizer and the classifier object
    #   together.
    return make_pipeline(TfidfVectorizer(analyzer ='char',ngram_range=(1, 3), min_df=3, max_df=0.9), LogisticRegression(solver='lbfgs'),)
