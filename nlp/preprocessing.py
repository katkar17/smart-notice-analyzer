"""
nlp/preprocessing.py
====================
PHASE 2: Text preprocessing + text statistics.

This file demonstrates the basic NLP preprocessing steps from the syllabus:
  1. Tokenization        - split text into words
  2. Lowercasing         - make everything lower case
  3. Stop-word removal   - drop common words like "the", "is", "of"
  4. Stemming            - chop words to their root (studies -> studi)
  5. Lemmatization       - reduce words to a real dictionary word (studies -> study)

It also calculates simple text statistics (word/sentence/char/unique counts).

We use the NLTK library. NLTK needs some small data files (tokenizer,
stop-word list, WordNet dictionary). The _ensure_nltk_data() function below
downloads them automatically the first time you run the app, so you don't
have to remember any extra command.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def _ensure_nltk_data():
    """
    Download the NLTK data packages we need, but only if they are missing.
    nltk.data.find() raises LookupError when a package is not installed yet,
    and that is our signal to download it.
    """
    needed = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),   # newer NLTK needs this too
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),          # dictionary used by the lemmatizer
        ("corpora/omw-1.4", "omw-1.4"),          # extra WordNet data
    ]
    for path, package in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


# Make sure the data is present as soon as this file is imported.
_ensure_nltk_data()

# Create these objects once (reusing them is faster than rebuilding each time).
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Run every preprocessing step on the given text and return a dictionary
    so the dashboard can display each stage separately.

    Returned keys:
      tokens              -> raw word tokens (only alphabetic words kept)
      lowercased          -> same words, lower case
      without_stopwords   -> stop-words removed
      stemmed             -> Porter stemmer applied
      lemmatized          -> WordNet lemmatizer applied
    """
    # 1. Tokenize into words, then keep only alphabetic tokens
    #    (this drops punctuation and standalone numbers like "." or ",").
    tokens = [t for t in word_tokenize(text) if t.isalpha()]

    # 2. Lowercasing
    lowercased = [t.lower() for t in tokens]

    # 3. Stop-word removal
    without_stopwords = [w for w in lowercased if w not in _stop_words]

    # 4. Stemming (fast, rule-based, can give non-real words)
    stemmed = [_stemmer.stem(w) for w in without_stopwords]

    # 5. Lemmatization (slower, uses a dictionary, gives real words)
    lemmatized = [_lemmatizer.lemmatize(w) for w in without_stopwords]

    return {
        "tokens": tokens,
        "lowercased": lowercased,
        "without_stopwords": without_stopwords,
        "stemmed": stemmed,
        "lemmatized": lemmatized,
    }


def get_text_statistics(text):
    """
    Calculate basic statistics about the notice:
      words        -> how many word tokens (alphabetic)
      sentences    -> how many sentences
      characters   -> total characters including spaces
      unique_words -> how many different words (case-insensitive)
    """
    words = [t for t in word_tokenize(text) if t.isalpha()]
    sentences = sent_tokenize(text)
    unique_words = set(w.lower() for w in words)

    return {
        "words": len(words),
        "sentences": len(sentences),
        "characters": len(text),
        "unique_words": len(unique_words),
    }
