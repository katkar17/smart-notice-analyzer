# VIVA Questions and Answers — Smart College Notice Analyzer

A set of common questions an examiner may ask, with short, simple answers.

---

**Q1. What is your project about?**
It is a web application that reads a college notice and uses NLP to show a
summary, keywords, named entities, statistics, a word cloud, and a simple
question-answering feature.

**Q2. What is NLP?**
Natural Language Processing is a branch of AI that helps computers understand,
process, and work with human language (text or speech).

**Q3. Which technologies did you use?**
Python and Flask for the backend; HTML, CSS, JavaScript and Bootstrap for the
frontend; and the NLP libraries NLTK, spaCy, scikit-learn, pypdf, and WordCloud.

**Q4. What is tokenization?**
Tokenization is splitting text into smaller pieces called tokens — usually
words or sentences. For example, "I study NLP" becomes ["I", "study", "NLP"].

**Q5. What are stop words and why remove them?**
Stop words are very common words like "the", "is", "of" that carry little
meaning. We remove them so the analysis focuses on the important words.

**Q6. What is the difference between stemming and lemmatization?**
Stemming chops a word to a rough root using rules (e.g. "studies" -> "studi"),
and the result may not be a real word. Lemmatization uses a dictionary to find
the proper base word (e.g. "studies" -> "study"). Lemmatization is more
accurate but slower.

**Q7. What is an N-gram?**
An N-gram is a sequence of N consecutive words. A unigram is 1 word, a bigram
is 2 words, and a trigram is 3 words. They help capture common phrases.

**Q8. What is TF-IDF?**
TF-IDF stands for Term Frequency – Inverse Document Frequency. It gives a word
a high score if it appears often in a document but is not common across all
documents. It helps identify the important words.

**Q9. How do you extract keywords?**
I split the notice into sentences, treat each as a small document, compute
TF-IDF scores with scikit-learn, add up each word's score, and take the top 10.

**Q10. What is Named Entity Recognition (NER)?**
NER finds and labels "named things" in text, such as people, organizations,
dates, places, money, and percentages. I use spaCy's `en_core_web_sm` model.

**Q11. Which entity types do you extract?**
PERSON, ORGANIZATION, DATE, LOCATION, MONEY, and PERCENT. spaCy's ORG is shown
as ORGANIZATION, and its GPE/LOC labels are shown as LOCATION.

**Q12. How does your summarizer work? Is it an AI/LLM?**
No LLM is used. It is extractive: I count word frequencies, score each
sentence by the sum of its word frequencies, and pick the top-scoring
sentences, keeping their original order.

**Q13. How does the question answering work?**
I convert each sentence and the question into TF-IDF vectors, then use cosine
similarity to find which sentence is most similar to the question. That
sentence is returned as the answer.

**Q14. What is cosine similarity?**
It measures the angle between two vectors. A value near 1 means they are very
similar; a value near 0 means they share little in common. It is used to
compare the question with each sentence.

**Q15. Why does "What is the last date?" sometimes not work?**
Because the matching is based on shared words. If the notice says "submitted
before 15 October" and never uses the word "date", the simple TF-IDF method
cannot connect them. This is a known limitation of word-based QA.

**Q16. How is the word cloud made?**
The WordCloud library counts word frequencies and draws more frequent words in
a larger size. The image is generated in memory and shown on the dashboard.

**Q17. What text statistics do you show?**
Number of words, number of sentences, number of characters, and the number of
unique words.

**Q18. How do you read PDF files?**
I use the pypdf library to open the PDF and extract the text from each page.
Only text-based PDFs are supported, not scanned images.

**Q19. Why Flask and not Django?**
Flask is lightweight and simple, which suits a small mini-project. Django is
larger and more suited to big applications.

**Q20. What are the limitations and future scope?**
Limitations: QA matches only on shared words, the summary is extractive, and
no OCR for scanned PDFs. Future scope: use word embeddings for smarter QA,
highlight the exact answer phrase, add OCR, and support comparing multiple
notices.

**Q21. What did you learn from this project?**
How to build a complete NLP pipeline — preprocessing, feature extraction
(TF-IDF), NER, summarization, and similarity-based QA — and connect it to a
simple, responsive web application that runs fully offline.

