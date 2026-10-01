"""
nlp/keywords.py
===============
PHASE 2: Keyword extraction (TF-IDF) + N-gram analysis.

Two syllabus concepts live here:

1. Keyword extraction using TF-IDF
   TF-IDF = Term Frequency x Inverse Document Frequency.
   It gives a high score to words that are frequent in the notice but
   not just common everywhere. We treat each SENTENCE as a small
   "document" so TF-IDF can tell important words from filler words.

2. N-gram analysis
   An N-gram is a group of N words in a row.
     unigram  = 1 word   ("examination")
     bigram   = 2 words  ("examination forms")
     trigram  = 3 words  ("semester examination forms")
   We count them and show the most frequent ones.
"""

from collections import Counter

from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text, top_n=10):
    """
    Return the top N keywords as a list of (word, score) tuples,
    highest score first.

    How it works:
      - Split the notice into sentences (each one is a "document").
      - TfidfVectorizer turns them into TF-IDF scores and removes
        English stop-words automatically.
      - We add up each word's score across all sentences and keep
        the highest ones.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Sum each word's TF-IDF score over every sentence.
    # .sum(axis=0) adds down the columns; .A1 flattens it to a 1-D list.
    summed_scores = tfidf_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    # Pair each word with its score and sort from high to low.
    word_scores = sorted(
        zip(words, summed_scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    # Round the score so it looks neat on the page.
    return [(word, round(float(score), 3)) for word, score in word_scores[:top_n]]


def get_ngrams(tokens, n, top_k=10):
    """
    Count the most frequent N-grams in a list of word tokens.

    Parameters:
      tokens -> list of words (we use the lower-cased words)
      n      -> 1 for unigrams, 2 for bigrams, 3 for trigrams
      top_k  -> how many of the most common ones to return

    Returns a list of (ngram_text, count) tuples.
    """
    if len(tokens) < n:
        return []

    all_ngrams = list(ngrams(tokens, n))
    counts = Counter(all_ngrams)

    # Join each tuple of words into a readable string: ("exam", "forms") -> "exam forms"
    return [(" ".join(gram), count) for gram, count in counts.most_common(top_k)]
