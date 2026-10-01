"""
Smart College Notice Analyzer
================================
PHASE 1: Flask setup + basic UI + notice input

What this file does (in plain words):
- Starts a small Flask web server.
- Shows a Home page where the student can paste a notice OR upload a .txt / .pdf file.
- Reads the text from whatever the student gave us.
- Sends that text to a Dashboard page.

In later phases we will add the NLP features (keywords, summary, NER,
word cloud, question answering, etc.) to the Dashboard. For now the
Dashboard just proves that the text was received correctly.
"""

import os
import base64
from io import BytesIO

from flask import Flask, render_template, request
from pypdf import PdfReader                 # used to read text from PDF files
from werkzeug.utils import secure_filename  # makes uploaded file names safe
from wordcloud import WordCloud             # Phase 3: generates the word cloud image

# Our own NLP helpers (from the nlp/ folder)
from nlp.preprocessing import preprocess_text, get_text_statistics
from nlp.keywords import extract_keywords, get_ngrams
from nlp.ner import extract_entities        # Phase 3
from nlp.summarizer import summarize        # Phase 3
from nlp.qa import answer_question          # Phase 4

# ---------------------------------------------------------------------------
# Basic Flask setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Folder where uploaded files are temporarily saved.
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Create the folder automatically if it does not exist yet.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# A ready-made sample notice. We show it on the Home page so the student
# can click one button and test the app immediately.
SAMPLE_NOTICE = (
    "University of Mumbai has announced that semester examination forms must "
    "be submitted before 15 October 2026. Students must have at least 75 "
    "percent attendance to be eligible for the examination. The examination "
    "will begin on 5 November 2026. Students should contact the examination "
    "department for further information."
)


# ---------------------------------------------------------------------------
# Helper functions to read uploaded files
# ---------------------------------------------------------------------------
def read_txt_file(file_path):
    """Read and return all text from a plain .txt file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(file_path):
    """Read and return all text from a .pdf file, page by page."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        # extract_text() can return None for empty pages, so we guard with "or ''"
        text += (page.extract_text() or "") + "\n"
    return text


def generate_wordcloud_image(text):
    """
    Create a word cloud from the text and return it as a base64 PNG string.
    We embed this string directly in the HTML page, so no image file needs
    to be saved on disk. Returns None if the text has no usable words.
    """
    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
        ).generate(text)
    except ValueError:
        # WordCloud raises ValueError if there are no words left to draw.
        return None

    image = wc.to_image()            # a PIL image
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    # Convert the raw PNG bytes into a text string the browser can read.
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes (the "pages" of our website)
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Home page: shows the paste box, the upload box, and the sample notice."""
    return render_template("index.html", sample=SAMPLE_NOTICE)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Runs when the student clicks "Analyze Notice".
    Steps:
      1. Read pasted text from the text box (if any).
      2. If a file was uploaded instead, read text from that file.
      3. If we still have no text, send the student back home with a message.
      4. Otherwise show the Dashboard with the collected text.
    """
    # 1. Text typed/pasted into the box.
    notice_text = request.form.get("notice_text", "").strip()

    # 2. A file uploaded through the form (may be empty).
    uploaded = request.files.get("notice_file")
    if uploaded and uploaded.filename:
        filename = secure_filename(uploaded.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded.save(save_path)

        lower_name = filename.lower()
        if lower_name.endswith(".txt"):
            notice_text = read_txt_file(save_path)
        elif lower_name.endswith(".pdf"):
            notice_text = read_pdf_file(save_path)
        else:
            return render_template(
                "index.html",
                sample=SAMPLE_NOTICE,
                error="Please upload only a .txt or .pdf file.",
            )

    # 3. Nothing to analyze.
    if not notice_text.strip():
        return render_template(
            "index.html",
            sample=SAMPLE_NOTICE,
            error="Please paste a notice or upload a .txt / .pdf file first.",
        )

    # 4. Run the Phase 2 NLP analysis on the collected text.
    processed = preprocess_text(notice_text)          # tokens, stemmed, lemmatized, ...
    statistics = get_text_statistics(notice_text)     # word/sentence/char counts
    keywords = extract_keywords(notice_text, top_n=10)  # TF-IDF top keywords

    # N-grams are built from the lower-cased words produced above.
    lower_words = processed["lowercased"]
    ngrams_result = {
        "unigrams": get_ngrams(lower_words, 1),
        "bigrams": get_ngrams(lower_words, 2),
        "trigrams": get_ngrams(lower_words, 3),
    }

    # Phase 3 analysis: named entities, summary, and word cloud.
    entities = extract_entities(notice_text)
    summary = summarize(notice_text, num_sentences=2)
    wordcloud_image = generate_wordcloud_image(notice_text)

    # Phase 4: if the student asked a question, find the best answer sentence.
    question = request.form.get("question", "").strip()
    qa_result = answer_question(notice_text, question) if question else None

    # 5. Show the dashboard with the real results.
    return render_template(
        "dashboard.html",
        notice_text=notice_text,
        processed=processed,
        statistics=statistics,
        keywords=keywords,
        ngrams=ngrams_result,
        entities=entities,
        summary=summary,
        wordcloud_image=wordcloud_image,
        question=question,
        qa_result=qa_result,
    )


# ---------------------------------------------------------------------------
# Start the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=True auto-reloads the server when you edit the code.
    app.run(debug=True)
