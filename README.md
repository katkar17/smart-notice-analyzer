# Smart College Notice Analyzer

An NLP mini-project: a small Flask web application that reads a college
notice (pasted text or an uploaded `.txt` / `.pdf` file) and analyzes it
using classic Natural Language Processing techniques. It produces a summary,
keywords, named entities, N-grams, text statistics, a word cloud, and a
simple question-answering feature — all running locally, with no AI/LLM API.

---

## 1. Title
Smart College Notice Analyzer

## 2. Problem Statement
College notices contain important information (deadlines, attendance rules,
exam dates) buried inside long paragraphs. Students often miss key details.
This project automatically reads a notice and highlights the important parts,
so a student can quickly understand what matters.

## 3. Objectives
- Accept a notice as pasted text or as a `.txt` / `.pdf` upload.
- Apply core NLP preprocessing (tokenization, stop-word removal, stemming, lemmatization).
- Extract keywords and the most frequent N-grams.
- Identify named entities such as dates, organizations, and locations.
- Produce a short extractive summary without using any LLM.
- Answer simple questions about the notice using text similarity.
- Show basic text statistics and a word cloud on a single dashboard.

## 4. Features
- Paste or upload a notice (`.txt` / `.pdf`).
- Text preprocessing with visible step-by-step output.
- Keyword extraction using TF-IDF (top 10).
- N-gram analysis (unigrams, bigrams, trigrams).
- Named Entity Recognition (PERSON, ORGANIZATION, DATE, LOCATION, MONEY, PERCENT).
- Extractive text summarization.
- Simple question answering (TF-IDF + cosine similarity).
- Word cloud image.
- Text statistics (words, sentences, characters, unique words).
- Clean, responsive dashboard built with Bootstrap.

## 5. Technologies
- Python 3
- Flask (web framework)
- HTML, CSS, JavaScript (front end)
- Bootstrap 5 (responsive UI)
- NLTK (tokenization, stop-words, stemming, lemmatization, summarization)
- spaCy (Named Entity Recognition)
- scikit-learn (TF-IDF, cosine similarity)
- pypdf (reading PDF files)
- WordCloud (word cloud image)

## 6. NLP Techniques Used
- Tokenization (splitting text into words and sentences)
- Lowercasing
- Stop-word removal
- Stemming (Porter stemmer)
- Lemmatization (WordNet)
- N-grams (unigram, bigram, trigram)
- TF-IDF (keyword extraction and question answering)
- Named Entity Recognition (spaCy)
- Cosine similarity (text similarity for question answering)
- Extractive summarization (word-frequency sentence scoring)
- Word cloud (frequency-based visualization)

## 7. System Flow
```
Home Page
   |
Paste notice  OR  upload .txt / .pdf
   |
Click "Analyze Notice"
   |
Flask reads the text  ->  nlp/ modules process it
   |
Dashboard shows: Summary, Keywords, Entities, Statistics,
N-grams, Preprocessing, Word Cloud, and Ask-a-Question
```

## 8. Installation
These steps assume Windows with Python 3.10+ installed.

```powershell
# 1. Go into the project folder
cd smart-notice-analyzer

# 2. Create a virtual environment
python -m venv venv

# 3. Install the dependencies (no need to "activate" the venv)
venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Download the spaCy English model (needed for Named Entity Recognition)
venv\Scripts\python.exe -m spacy download en_core_web_sm
```

The NLTK data files (tokenizer, stop-words, WordNet) download automatically
the first time you analyze a notice. If your network blocks that, run:

```powershell
venv\Scripts\python.exe -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

## 9. How to Run
```powershell
venv\Scripts\python.exe app.py
```
Then open the printed address (usually `http://127.0.0.1:5000`) in a browser.

## 10. Sample Input
> University of Mumbai has announced that semester examination forms must be
> submitted before 15 October 2026. Students must have at least 75 percent
> attendance to be eligible for the examination. The examination will begin on
> 5 November 2026. Students should contact the examination department for
> further information.

## 11. Sample Output
- Summary: the two most important sentences (exam-form deadline and attendance rule).
- Keywords: examination, students, attendance, forms, percent, ...
- Named Entities: ORGANIZATION = University of Mumbai; DATE = 15 October 2026, 5 November 2026; PERCENT = 75 percent.
- Statistics: words, sentences, characters, and unique-word counts.
- Word Cloud: "examination" shown largest.
- Question "What is the attendance requirement?" ->
  "Students must have at least 75 percent attendance to be eligible for the examination."

## 12. Limitations
- The question answering matches on shared words (TF-IDF + cosine similarity),
  so it works best when the question uses words that appear in the notice.
  It does not understand synonyms, so "What is the last date?" may not match a
  sentence that says "submitted before 15 October" because the word "date" is
  not in that sentence.
- The summary is extractive (it selects existing sentences); it does not
  rewrite or paraphrase.
- Entity accuracy depends on the small spaCy model (`en_core_web_sm`).
- Scanned/image PDFs are not supported (no OCR); only text-based PDFs work.
- Designed for short notices, not large documents.

## 13. Future Scope
- Add synonym handling or word embeddings for smarter question answering.
- Highlight the answer phrase (e.g. the exact date) inside the sentence.
- Support more file formats and OCR for scanned PDFs.
- Allow analyzing and comparing multiple notices.
- Add a download/report option for the analysis.

## 14. Conclusion
This project demonstrates the core NLP pipeline — preprocessing, N-grams,
keyword extraction, named entity recognition, summarization, text similarity,
question answering, and visualization — inside a small, easy-to-understand
Flask web application. It is a practical way to connect NLP theory from the
syllabus to a working, local application.

