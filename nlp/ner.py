"""
nlp/ner.py
==========
PHASE 3: Named Entity Recognition (NER) using spaCy.

NER finds "named things" in text and labels them. We extract the six
categories the project asks for:
    PERSON, ORGANIZATION, DATE, LOCATION, MONEY, PERCENT

spaCy uses slightly different label names, so we translate them:
    spaCy ORG         -> ORGANIZATION
    spaCy GPE / LOC   -> LOCATION
The others (PERSON, DATE, MONEY, PERCENT) are already named the same.

IMPORTANT: spaCy needs a language model called "en_core_web_sm".
Install it once with:
    python -m spacy download en_core_web_sm
"""

# We load the spaCy model only when it is first needed ("lazy loading").
# This keeps the app fast to start and gives a clear error message if the
# model has not been downloaded yet.
_nlp = None

# Which spaCy labels we keep, and what we rename them to for display.
_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",      # GPE = countries, cities, states
    "LOC": "LOCATION",      # LOC = other places (rivers, mountains, etc.)
    "DATE": "DATE",
    "MONEY": "MONEY",
    "PERCENT": "PERCENT",
}


def _get_nlp():
    """Load the spaCy model once and reuse it on later calls."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Raised when the model is not installed. Give a helpful message.
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Run this command once:\n"
                "    python -m spacy download en_core_web_sm"
            )
    return _nlp


def extract_entities(text):
    """
    Return a dictionary mapping each category to a list of unique entities.

    Example:
        {
          "PERSON": [],
          "ORGANIZATION": ["University of Mumbai"],
          "DATE": ["15 October 2026", "5 November 2026"],
          "LOCATION": [],
          "MONEY": [],
          "PERCENT": ["at least 75 percent"],
        }
    """
    nlp = _get_nlp()
    doc = nlp(text)

    # Start with an empty list for every category.
    results = {name: [] for name in ["PERSON", "ORGANIZATION", "DATE",
                                     "LOCATION", "MONEY", "PERCENT"]}

    for ent in doc.ents:
        if ent.label_ in _LABEL_MAP:
            category = _LABEL_MAP[ent.label_]
            entity_text = ent.text.strip()
            # Avoid adding the same entity twice.
            if entity_text not in results[category]:
                results[category].append(entity_text)

    return results
