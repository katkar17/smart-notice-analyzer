"""
nlp/qa.py
=========
PHASE 4: Simple Question Answering (QA) using TF-IDF + cosine similarity.

Idea (NO AI / NO LLM):
  - Treat each sentence of the notice as a possible answer.
  - Turn the sentences AND the question into TF-IDF vectors (numbers).
  - Use cosine similarity to measure how "close" the question is to each
    sentence. Cosine similarity = 1.0 means very similar, 0.0 means no
    shared important words.
  - Return the sentence with the highest similarity as the answer.

Because it matches on SHARED WORDS, the QA works best when your question
uses words that actually appear in the notice (e.g. ask "attendance" if the
notice says "attendance").
"""

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def answer_question(text, question):
    """
    Find the sentence in `text` most relevant to `question`.

    Returns a dictionary:
        {"sentence": <best sentence or None>, "score": <similarity 0..1>}
    If no sentence shares any important word with the question, sentence
    is None.
    """
    sentences = sent_tokenize(text)
    if not sentences or not question.strip():
        return {"sentence": None, "score": 0.0}

    # Build TF-IDF vectors. We fit on the notice sentences, then transform
    # the question using the SAME vocabulary so they can be compared.
    vectorizer = TfidfVectorizer(stop_words="english")
    sentence_vectors = vectorizer.fit_transform(sentences)
    question_vector = vectorizer.transform([question])

    # Cosine similarity between the question and every sentence.
    # The result is one score per sentence.
    similarities = cosine_similarity(question_vector, sentence_vectors)[0]

    best_index = similarities.argmax()      # position of the highest score
    best_score = float(similarities[best_index])

    # A score of 0 means the question shared no important words with any
    # sentence, so we have no good answer.
    if best_score == 0:
        return {"sentence": None, "score": 0.0}

    return {"sentence": sentences[best_index], "score": round(best_score, 3)}
