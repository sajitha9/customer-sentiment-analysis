# 😊 Customer Sentiment Analysis — End-to-End ML Project

A complete, beginner-friendly machine learning project that classifies
product reviews as **Positive**, **Negative**, or **Neutral** using
NLP + Logistic Regression, wrapped in an interactive Streamlit web app.

---

## 📁 Project Structure

```
sentiment_project/
│
├── app.py              ← Main Streamlit web application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🚀 How to Run (Step by Step)

### Step 1 — Create & activate a virtual environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Launch the app
```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 🧠 What This Project Covers

| Concept | Where Used |
|---|---|
| Text Cleaning (regex, lowercase) | `clean_text()` function |
| Stopword Removal | NLTK stopwords |
| Stemming | PorterStemmer |
| TF-IDF Vectorization | Sklearn TfidfVectorizer |
| Logistic Regression | Sklearn LogisticRegression |
| ML Pipeline | Sklearn Pipeline |
| Model Evaluation | Accuracy, F1, Confusion Matrix |
| Model Persistence | pickle save/load |
| Web UI | Streamlit |

---

## 🔧 ML Pipeline (How It Works)

```
Raw Review Text
      │
      ▼
  Text Cleaning
  (lowercase, remove URLs, special chars)
      │
      ▼
  Stopword Removal + Stemming
  ("running" → "run", remove "the", "is"...)
      │
      ▼
  TF-IDF Vectorization
  (convert words to numbers, bigrams, 5000 features)
      │
      ▼
  Logistic Regression Classifier
      │
      ▼
  Prediction + Confidence Score
  (Positive 92%, Negative 5%, Neutral 3%)
```

---

## 🎯 App Features

### Tab 1 — Analyze Single Review
- Type any product review OR choose a sample
- Instantly get sentiment + confidence score
- Visual confidence bar chart
- See pre-processed text

### Tab 2 — Bulk Analysis
- Upload a CSV file (must have a `review` column)
- OR paste multiple reviews (one per line)
- Get results table + distribution chart
- Download results as CSV

### Tab 3 — Model Insights
- Accuracy, F1 score metrics
- Confusion matrix visualization
- Training data distribution
- How the model works (explained simply)

---

## 📊 Model Details

- **Algorithm:** Logistic Regression
- **Features:** TF-IDF with bigrams (max 5000 features)
- **Classes:** Positive, Negative, Neutral
- **Built-in Dataset:** 55 sample reviews (no download needed)
- **Model saved as:** `sentiment_model.pkl` (auto-created on first run)

---

## 🔄 How to Use Your Own Dataset

Replace the `get_sample_data()` function with your own CSV:

```python
@st.cache_data
def get_sample_data():
    df = pd.read_csv("your_reviews.csv")
    # Make sure it has 'review' and 'sentiment' columns
    # sentiment values: 'positive', 'negative', 'neutral'
    return df
```

**Popular free datasets to try:**
- [IMDb Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Amazon Product Reviews](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- [Yelp Reviews](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)

---

## ⬆️ Upgrade Ideas (Next Steps)

1. **Use HuggingFace Transformers** — swap Logistic Regression for BERT/DistilBERT
2. **Add more classes** — Very Positive, Very Negative (5-star scale)
3. **Aspect-based sentiment** — which product features are positive/negative
4. **Deploy online** — push to [Streamlit Cloud](https://streamlit.io/cloud) for free hosting
5. **Add more data** — pull real reviews from a Kaggle dataset

---

## 💡 Sample CSV Format for Bulk Upload

```csv
review
"Amazing product, highly recommend!"
"Terrible quality, broke after one day."
"It's okay, nothing special."
"Love it! Works perfectly."
"Very disappointed with this purchase."
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| NLTK error | Run `python -c "import nltk; nltk.download('stopwords')"` |
| Port already in use | Run `streamlit run app.py --server.port 8502` |
| Old model loading wrong | Delete `sentiment_model.pkl` and rerun |

---

Happy Learning! 🎓
