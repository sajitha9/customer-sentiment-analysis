import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Download NLTK data ──────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Sentiment Analyzer",
    page_icon="😊",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { font-size: 16px; }
    .sentiment-box {
        padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
        margin: 10px 0;
    }
    .positive  { background: #d4edda; color: #155724; border: 2px solid #28a745; }
    .negative  { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .neutral   { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .metric-card {
        background: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Text preprocessing ───────────────────────────────────────────────────────
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words("english")) - {"not", "no", "never", "but", "very"}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# ── Sample dataset (built-in so no download needed) ─────────────────────────
@st.cache_data
def get_sample_data():
    positive_reviews = [
        "This product is absolutely amazing! Best purchase I've ever made.",
        "Excellent quality, fast shipping, very happy with my order.",
        "I love this! It exceeded all my expectations.",
        "Fantastic product, works perfectly. Highly recommend to everyone.",
        "Great value for money. The quality is outstanding.",
        "Very impressed with the build quality. Will buy again for sure.",
        "Perfect product, exactly as described. Super fast delivery too.",
        "Amazing customer service and the product is top notch.",
        "This is exactly what I needed. Works like a charm!",
        "Five stars! Brilliant product, arrived on time, well packaged.",
        "Outstanding quality! Really happy with this purchase.",
        "Superb! Does exactly what it says. Very pleased.",
        "Wonderful experience from start to finish. Product is excellent.",
        "Best product in its category. Totally worth every penny.",
        "Impressed with the quality. Delivery was super fast as well.",
        "Love this product! Using it every day and it's perfect.",
        "Great purchase. My whole family loves it.",
        "Top quality, great price. Would definitely buy again.",
        "Really happy with this. Solid and well made.",
        "Exceeded my expectations. Fast shipping, great packaging.",
    ]
    negative_reviews = [
        "Terrible product, broke after just two days of use.",
        "Very disappointed. The quality is extremely poor.",
        "Waste of money. Nothing like what was advertised.",
        "Awful experience. Customer service was useless and rude.",
        "Product arrived damaged and the seller refused to help.",
        "Does not work at all. Complete garbage.",
        "Horrible quality. Fell apart immediately after opening.",
        "Never buying from this brand again. Totally ruined.",
        "Extremely disappointed with this purchase. Zero stars.",
        "Poor packaging, product arrived broken. Requesting refund.",
        "Worst product I have ever bought. Avoid at all costs.",
        "Defective item sent. Still waiting for a replacement.",
        "Very bad quality. Material feels cheap and flimsy.",
        "Doesn't match the description at all. Misleading listing.",
        "Useless product. Didn't work from day one.",
        "Terrible customer support. They ignored my complaint.",
        "Not happy at all. The product stopped working after a week.",
        "Would not recommend. Cheap build, breaks easily.",
        "Completely useless. Waste of time and money.",
        "Received wrong item. No response from seller.",
    ]
    neutral_reviews = [
        "The product is okay, nothing special but gets the job done.",
        "Average quality, works as expected for the price.",
        "It's decent. Not the best but not the worst either.",
        "Delivery was on time. Product is average quality.",
        "It does what it's supposed to, nothing more nothing less.",
        "Acceptable product. A few minor issues but mostly fine.",
        "Product is fine. Not super impressed but not disappointed.",
        "Standard quality. Meets basic requirements.",
        "Reasonable price for what you get.",
        "Product works, just nothing exceptional about it.",
        "It's okay. Looks different from the photo but usable.",
        "Neutral experience. Would consider other options next time.",
        "Normal product. Does the job adequately.",
        "Neither good nor bad. Just an ordinary purchase.",
        "Works fine for basic use. Nothing to rave about.",
    ]
    reviews = positive_reviews + negative_reviews + neutral_reviews
    labels  = (["positive"] * len(positive_reviews) +
               ["negative"] * len(negative_reviews) +
               ["neutral"]  * len(neutral_reviews))
    df = pd.DataFrame({"review": reviews, "sentiment": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Train / load model ───────────────────────────────────────────────────────
MODEL_PATH = "sentiment_model.pkl"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    df = get_sample_data()
    df["clean"] = df["review"].apply(clean_text)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )
    pipeline.fit(X_train, y_train)
    preds   = pipeline.predict(X_test)
    acc     = accuracy_score(y_test, preds)
    report  = classification_report(y_test, preds, output_dict=True)
    cm      = confusion_matrix(y_test, preds, labels=["positive", "negative", "neutral"])

    result = {"pipeline": pipeline, "accuracy": acc, "report": report,
              "cm": cm, "X_test": X_test, "y_test": y_test}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(result, f)
    return result

# ── Emoji & colour helpers ───────────────────────────────────────────────────
EMOJI = {"positive": "😊 Positive", "negative": "😞 Negative", "neutral": "😐 Neutral"}
CSS   = {"positive": "positive",    "negative": "negative",    "neutral": "neutral"}

def confidence_bar(probs, classes):
    fig, ax = plt.subplots(figsize=(6, 2))
    colours = {"positive": "#28a745", "negative": "#dc3545", "neutral": "#ffc107"}
    bars = ax.barh(classes, probs, color=[colours[c] for c in classes], height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Confidence")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

# ── Main App ─────────────────────────────────────────────────────────────────
def main():
    st.title("🛍️ Customer Sentiment Analysis")
    st.markdown("**Real-time AI model** that classifies product reviews as Positive, Negative, or Neutral")
    st.markdown("---")

    model_data = load_or_train_model()
    pipeline   = model_data["pipeline"]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("Model Accuracy", f"{model_data['accuracy']:.1%}")
        st.markdown("**Per-class F1 Scores**")
        report = model_data["report"]
        for cls in ["positive", "negative", "neutral"]:
            f1 = report[cls]["f1-score"]
            st.progress(f1, text=f"{cls.capitalize()}: {f1:.2f}")
        st.markdown("---")
        st.markdown("**Algorithm:** Logistic Regression + TF-IDF")
        st.markdown("**Features:** Bigrams, stemming, stopword removal")
        st.markdown("**Dataset:** 55 sample reviews (3 classes)")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Review", "📦 Bulk Analysis", "📈 Model Insights"])

    # ── Tab 1: Single review ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Analyze a Single Review")
        sample_options = {
            "— choose a sample —": "",
            "😊 Happy customer": "This product is absolutely amazing! Best purchase ever. Fast delivery!",
            "😞 Unhappy customer": "Terrible quality, broke after two days. Complete waste of money.",
            "😐 Neutral customer": "Product is okay, nothing special but gets the job done.",
            "🤔 Mixed feelings": "The product is good but the delivery was very slow and packaging was poor.",
        }
        choice = st.selectbox("Try a sample review:", list(sample_options.keys()))
        default_text = sample_options[choice]

        review_text = st.text_area(
            "Or type your own review:",
            value=default_text,
            height=130,
            placeholder="Type a product review here...",
        )

        if st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary"):
            if review_text.strip():
                cleaned   = clean_text(review_text)
                pred      = pipeline.predict([cleaned])[0]
                probs     = pipeline.predict_proba([cleaned])[0]
                classes   = pipeline.classes_

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(
                        f'<div class="sentiment-box {CSS[pred]}">{EMOJI[pred]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Confidence:** {max(probs):.1%}")
                    st.markdown(f"**Word count:** {len(review_text.split())}")
                with col2:
                    st.pyplot(confidence_bar(probs, classes))

                with st.expander("🔧 Pre-processed text (what the model sees)"):
                    st.code(cleaned)
            else:
                st.warning("Please enter a review to analyze.")

    # ── Tab 2: Bulk CSV ──────────────────────────────────────────────────────
    with tab2:
        st.subheader("Analyze Multiple Reviews at Once")
        st.info("Upload a CSV with a column named **review** OR paste multiple reviews (one per line).")

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        pasted   = st.text_area("Or paste reviews (one per line):", height=150,
                                placeholder="Great product!\nTerrible quality.\nOkay for the price.")

        if st.button("🚀 Analyze All Reviews", use_container_width=True, type="primary"):
            reviews = []
            if uploaded:
                df_up = pd.read_csv(uploaded)
                if "review" in df_up.columns:
                    reviews = df_up["review"].dropna().tolist()
                else:
                    st.error("CSV must have a column named 'review'.")
            elif pasted.strip():
                reviews = [r.strip() for r in pasted.strip().split("\n") if r.strip()]

            if reviews:
                cleaned_reviews = [clean_text(r) for r in reviews]
                preds  = pipeline.predict(cleaned_reviews)
                probs  = pipeline.predict_proba(cleaned_reviews).max(axis=1)

                df_results = pd.DataFrame({
                    "Review":     reviews,
                    "Sentiment":  preds,
                    "Confidence": [f"{p:.1%}" for p in probs],
                })
                st.dataframe(df_results, use_container_width=True, height=300)

                counts = pd.Series(preds).value_counts()
                fig, ax = plt.subplots(figsize=(5, 3))
                colours = [("#28a745" if l == "positive" else
                            "#dc3545" if l == "negative" else "#ffc107")
                           for l in counts.index]
                ax.bar(counts.index, counts.values, color=colours)
                ax.set_title("Sentiment Distribution")
                ax.set_ylabel("Count")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

                csv_out = df_results.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results CSV", csv_out,
                                   "sentiment_results.csv", "text/csv")
            else:
                st.warning("No reviews to analyze.")

    # ── Tab 3: Model insights ────────────────────────────────────────────────
    with tab3:
        st.subheader("Model Performance & Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><h2>🎯</h2>'
                        f'<h3>{model_data["accuracy"]:.1%}</h3><p>Overall Accuracy</p></div>',
                        unsafe_allow_html=True)
        with col2:
            avg_f1 = np.mean([model_data["report"][c]["f1-score"]
                              for c in ["positive", "negative", "neutral"]])
            st.markdown('<div class="metric-card"><h2>📊</h2>'
                        f'<h3>{avg_f1:.2f}</h3><p>Avg F1 Score</p></div>',
                        unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h2>🏷️</h2>'
                        '<h3>3</h3><p>Sentiment Classes</p></div>',
                        unsafe_allow_html=True)

        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=model_data["cm"],
            display_labels=["Positive", "Negative", "Neutral"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix (Test Set)")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Sample Data Distribution")
        df_sample = get_sample_data()
        counts    = df_sample["sentiment"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        colours   = {"positive": "#28a745", "negative": "#dc3545", "neutral": "#ffc107"}
        ax2.bar(counts.index, counts.values,
                color=[colours[l] for l in counts.index])
        ax2.set_title("Training Data Distribution")
        ax2.set_ylabel("Number of Reviews")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)

        with st.expander("📖 How the model works"):
            st.markdown("""
**Pipeline steps:**
1. **Text Cleaning** — lowercase, remove URLs & special chars
2. **Stopword Removal** — remove common words (but keep "not", "no", "never")
3. **Stemming** — reduce words to root form (e.g., "running" → "run")
4. **TF-IDF Vectorization** — convert text to numeric features (bigrams, max 5000 features)
5. **Logistic Regression** — classify into 3 sentiment classes

**Why Logistic Regression?**
- Fast, interpretable, works well with TF-IDF
- Provides calibrated probabilities (confidence scores)
- Great baseline for text classification tasks
""")

if __name__ == "__main__":
    main()
