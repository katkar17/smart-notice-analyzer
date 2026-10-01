"""
nlp/summarizer.py
=================
PHASE 3: Simple extractive text summarization (NO AI / NO LLM).

"Extractive" means we pick the most important sentences from the original
text and join them together, rather than writing new sentences.

The algorithm (classic word-frequency method):
  1. Split the text into sentences.
  2. Count how often each (non stop-word) word appears.
  3. Give each sentence a score = sum of its word frequencies.
  4. Pick the highest-scoring sentences.
  5. Put them back in their original order so the summary reads naturally.
"""

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

_stop_words = set(stopwords.words("english"))


def summarize(text, num_sentences=2):
    """
    Return a short summary made of the `num_sentences` most important
    sentences from `text`.
    """
    sentences = sent_tokenize(text)

    # If the text is already short, just return it unchanged.
    if len(sentences) <= num_sentences:
        return text

    # Step 2: build a word-frequency table (skip stop-words and punctuation).
    frequencies = {}
    for word in word_tokenize(text.lower()):
        if word.isalpha() and word not in _stop_words:
            frequencies[word] = frequencies.get(word, 0) + 1

    if not frequencies:
        return text

    # Normalize so the most common word has a weight of 1.0.
    highest = max(frequencies.values())
    for word in frequencies:
        frequencies[word] = frequencies[word] / highest

    # Step 3: score each sentence by adding up its word weights.
    sentence_scores = {}
    for index, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in frequencies:
                sentence_scores[index] = sentence_scores.get(index, 0) + frequencies[word]

    # Step 4: choose the indexes of the top-scoring sentences.
    top_indexes = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    top_indexes = top_indexes[:num_sentences]

    # Step 5: sort those indexes so sentences stay in their original order.
    top_indexes = sorted(top_indexes)

    return " ".join(sentences[i] for i in top_indexes)
