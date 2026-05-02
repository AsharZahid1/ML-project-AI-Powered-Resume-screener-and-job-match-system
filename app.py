"""
AI-Powered Resume Screener & Job Match System
=============================================
Models  : SVM, Naive Bayes, Logistic Regression, Decision Tree, KNN
UI      : Gradio
Vectors : TF-IDF
Dataset : Kaggle Resume Dataset (UpdatedResumedataset.csv)
"""

import re
import pickle
import os
import warnings
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  SKILL KEYWORD BANK
# ──────────────────────────────────────────────

SKILL_KEYWORDS = [
    "python","java","javascript","typescript","c++","c#","r","scala","go",
    "kotlin","swift","php","ruby","matlab","machine learning","deep learning",
    "nlp","computer vision","tensorflow","pytorch","keras","scikit-learn",
    "bert","gpt","neural network","reinforcement learning","pandas","numpy",
    "sql","mysql","postgresql","mongodb","hadoop","spark","tableau","power bi",
    "data analysis","data visualization","feature engineering","etl","django",
    "flask","fastapi","node.js","react","angular","vue","rest api","graphql",
    "docker","kubernetes","aws","azure","google cloud","git","linux","bash",
    "agile","scrum","communication","teamwork","leadership","opencv","gradio",
    "streamlit","excel","powerpoint","statistics","probability","regression",
    "classification","clustering","svm","naive bayes","decision tree","knn",
    "random forest","xgboost","logistic regression","tfidf","nltk","spacy"
]

# ──────────────────────────────────────────────
#  TEXT CLEANING
# ──────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# ──────────────────────────────────────────────
#  TRAIN ALL 5 MODELS
# ──────────────────────────────────────────────

models = {}
vectorizer = None
label_encoder = None
model_accuracies = {}

def train_models():
    global models, vectorizer, label_encoder, model_accuracies

    print("\n📂 Loading dataset...")

    # Try to load dataset — supports both dataset formats
    dataset_path = None
    for candidate in ["resume_dataset.csv", "Resume.csv", "UpdatedResumedataset.csv"]:
        if os.path.exists(candidate):
            dataset_path = candidate
            break

    if dataset_path is None:
        # Fallback: synthetic data so app still launches without dataset
        print("⚠️  Dataset not found — using synthetic fallback data.")
        print("    Place your dataset CSV (resume_dataset.csv) in the same folder as app.py")
        print("    Expected columns: 'Category' and 'Resume_str' (or 'Resume')")
        categories = [
            "Information-Technology", "HR", "Designer", "Teacher",
            "Healthcare", "Finance", "Sales", "Engineering",
            "Business-Development", "Accountant"
        ]
        samples = [
            "python machine learning data analysis pandas numpy scikit-learn statistics regression classification",
            "recruitment hiring onboarding employee relations communication leadership hr",
            "photoshop illustrator figma ui ux design graphic adobe creative",
            "teaching curriculum lesson plan classroom management education",
            "nursing patient care medical clinical hospital health",
            "accounting finance budgeting tax audit excel financial reporting",
            "sales crm customer relationship management business development excel",
            "java python c++ software engineering algorithms data structures",
            "business strategy market analysis operations management consulting",
            "bookkeeping accounts payable receivable ledger balance sheet"
        ]
        rows = []
        for cat, sample in zip(categories, samples):
            for _ in range(60):
                noise = " ".join(np.random.choice(sample.split(), size=25, replace=True))
                rows.append({"Category": cat, "Resume_str": noise})
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(dataset_path)
        print(f"   Columns found: {list(df.columns)}")

    # ── Normalize column names ──
    # Support both 'Resume_str' (new dataset) and 'Resume' (old dataset)
    if 'Resume_str' in df.columns:
        df['Resume'] = df['Resume_str']
    elif 'Resume' not in df.columns:
        # Try to find a text column automatically
        text_cols = [c for c in df.columns if df[c].dtype == object and c != 'Category']
        if text_cols:
            df['Resume'] = df[text_cols[0]]
            print(f"   Using column '{text_cols[0]}' as resume text.")
        else:
            raise ValueError("Could not find resume text column. Expected 'Resume_str' or 'Resume'.")

    # Drop rows with missing values
    df = df[['Category', 'Resume']].dropna()

    print(f"✅ Dataset loaded: {len(df)} resumes, {df['Category'].nunique()} categories")
    print(f"   Categories: {sorted(df['Category'].unique())}")

    # Clean
    df['cleaned'] = df['Resume'].apply(clean_text)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Category'])

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['cleaned'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define all 5 models
    classifiers = {
        "SVM":                 SVC(kernel='linear', probability=True, random_state=42),
        "Naive Bayes":         MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }

    print("\n🏋️  Training all 5 models...")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        models[name] = clf
        model_accuracies[name] = round(acc, 2)
        print(f"   {name:25s} → Accuracy: {acc:.2f}%")

    print("\n✅ All models trained!\n")

# ──────────────────────────────────────────────
#  PDF EXTRACTION
# ──────────────────────────────────────────────

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file.name) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return f"ERROR: {str(e)}"
    return text.strip()

# ──────────────────────────────────────────────
#  SKILL EXTRACTION
# ──────────────────────────────────────────────

def extract_skills(text):
    text_lower = text.lower()
    found = set()
    for skill in SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill.title())
    return found

# ──────────────────────────────────────────────
#  ACCURACY CHART
# ──────────────────────────────────────────────

def make_accuracy_chart():
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0e0e1a')
    ax.set_facecolor('#0e0e1a')

    names = list(model_accuracies.keys())
    accs = list(model_accuracies.values())
    best = max(model_accuracies, key=model_accuracies.get)
    colors = ['#f7c948' if n == best else '#7c6af7' for n in names]

    bars = ax.barh(names, accs, color=colors, height=0.55, edgecolor='none')
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)", color='#aaaacc', fontsize=10)
    ax.set_title(f"Model Accuracy Comparison  (🏆 Best: {best} {model_accuracies[best]}%)", color='white', fontsize=11, pad=12)
    ax.tick_params(colors='#aaaacc')
    ax.spines[['top','right','bottom','left']].set_visible(False)

    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc}%', va='center', color='white', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────
#  MAIN PREDICT FUNCTION
# ──────────────────────────────────────────────

def analyze_resume(pdf_file, job_description, selected_model):
    if pdf_file is None:
        return "⚠️ Please upload a resume PDF.", "", "", "", "", None

    if not job_description.strip():
        return "⚠️ Please paste a job description.", "", "", "", "", None

    # Extract resume text
    resume_text = extract_text_from_pdf(pdf_file)
    if resume_text.startswith("ERROR"):
        return resume_text, "", "", "", "", None

    cleaned_resume = clean_text(resume_text)

    # ── Job Category Prediction ──
    resume_vec = vectorizer.transform([cleaned_resume])
    model = models[selected_model]
    pred_label = model.predict(resume_vec)[0]
    predicted_category = label_encoder.inverse_transform([pred_label])[0]

    # Confidence (if model supports it)
    try:
        proba = model.predict_proba(resume_vec)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [(label_encoder.inverse_transform([i])[0], round(proba[i]*100, 1)) for i in top3_idx]
        confidence_str = "\n".join([f"  {rank+1}. {cat} — {conf}%" for rank, (cat, conf) in enumerate(top3)])
    except:
        confidence_str = f"  Predicted: {predicted_category}"

    # ── Cosine Similarity Score ──
    job_vec = vectorizer.transform([clean_text(job_description)])
    cos_score = float(cosine_similarity(resume_vec, job_vec)[0][0]) * 100
    cos_score = round(cos_score, 2)

    if cos_score >= 70:
        score_label = "🟢 Excellent Match"
    elif cos_score >= 45:
        score_label = "🟡 Good Match"
    elif cos_score >= 25:
        score_label = "🟠 Partial Match"
    else:
        score_label = "🔴 Low Match"

    # ── Skill Analysis ──
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    extra = resume_skills - job_skills

    # ── Recommendation ──
    if cos_score >= 70:
        verdict = f"✅ Strong match! This resume aligns well with the job description. Predicted role category: {predicted_category}."
    elif cos_score >= 45:
        verdict = f"🔍 Moderate match. The resume fits '{predicted_category}' but has some skill gaps. Consider adding: {', '.join(list(missing)[:4]) or 'none detected'}."
    elif cos_score >= 25:
        verdict = f"⚠️ Partial match. Resume is classified as '{predicted_category}'. Significant gaps detected. Focus on: {', '.join(list(missing)[:5]) or 'domain skills'}."
    else:
        verdict = f"❌ Low match. Resume ({predicted_category}) doesn't align well with this job. Major reskilling recommended."

    # ── Best model recommendation ──
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_accuracy = model_accuracies[best_model_name]
    if selected_model == best_model_name:
        best_note = f"🏆 You are using the best model: {best_model_name} ({best_accuracy}%)"
    else:
        best_note = (
            f"🏆 Best Model: {best_model_name} ({best_accuracy}%)\n"
            f"Currently using: {selected_model} ({model_accuracies[selected_model]}%)\n"
            f"💡 Switch to {best_model_name} for highest accuracy."
        )

    # Accuracy chart
    chart = make_accuracy_chart()

    return (
        f"{cos_score}%  —  {score_label}\n(Based on TF-IDF cosine similarity — same for all models)",
        f"Predicted Category: {predicted_category}\n\nTop Predictions:\n{confidence_str}\n\n{best_note}",
        ", ".join(sorted(matched)) or "No direct matches found.",
        ", ".join(sorted(missing)) or "✅ No critical skill gaps!",
        verdict,
        chart
    )

# ──────────────────────────────────────────────
#  GRADIO UI
# ──────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif !important; }

body, .gradio-container {
    background: #080810 !important;
    color: #dde !important;
}

.gradio-container { max-width: 1150px !important; margin: 0 auto !important; }

.header {
    text-align: center;
    padding: 2.5rem 1rem 1.8rem;
    border-bottom: 1px solid #1c1c2e;
    margin-bottom: 1.5rem;
}
.header h1 {
    font-size: 2.2rem; font-weight: 700; margin: 0 0 0.4rem;
    background: linear-gradient(135deg, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header p { color: #777799; font-size: 0.95rem; margin: 0; }

.badge {
    display: inline-block; background: #1a1a2e; border: 1px solid #2a2a42;
    border-radius: 999px; padding: 0.2rem 0.8rem; font-size: 0.78rem;
    color: #9999bb; margin: 0.5rem 0.2rem 0;
}

label span {
    color: #8888bb !important; font-size: 0.8rem !important;
    font-weight: 600 !important; letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

textarea, .gr-file-upload {
    background: #0e0e1c !important;
    border: 1px solid #252538 !important;
    border-radius: 10px !important;
    color: #dde !important;
}

textarea:focus { border-color: #a78bfa !important; }

.gr-button-primary {
    background: linear-gradient(135deg, #a78bfa, #34d399) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 1rem !important;
    color: white !important; padding: 0.7rem !important;
}
"""

# ── Train on startup ──
train_models()

with gr.Blocks(css=CSS, title="AI Resume Screener") as demo:

    gr.HTML("""
    <div class="header">
        <h1>🤖 AI Resume Screener</h1>
        <p>Upload Resume · Paste Job Description · Compare 5 ML Models</p>
        <div>
            <span class="badge">SVM</span>
            <span class="badge">Naive Bayes</span>
            <span class="badge">Logistic Regression</span>
            <span class="badge">Decision Tree</span>
            <span class="badge">KNN</span>
        </div>
    </div>
    """)

    with gr.Row():
        # ── Left Column: Inputs ──
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📄 Upload Resume (PDF)",
                file_types=[".pdf"]
            )
            job_input = gr.Textbox(
                label="📋 Paste Job Description",
                placeholder="Paste the full job description here...",
                lines=10
            )
            model_selector = gr.Dropdown(
                choices=["SVM", "Naive Bayes", "Logistic Regression", "Decision Tree", "KNN"],
                value="SVM",
                label="🧠 Select ML Model — Switch to compare predictions"
            )
            gr.HTML("""
            <div style="background:#0e0e1c;border:1px solid #2a2a40;border-radius:8px;
                        padding:0.65rem 1rem;font-size:0.8rem;color:#8888bb;margin-top:-8px;">
                ℹ️ <strong style="color:#a78bfa;">Match Score & Skills</strong> are the same for all models
                (TF-IDF cosine similarity). Only the <strong style="color:#34d399;">Predicted Category</strong>
                changes when you switch models.
            </div>
            """)
            analyze_btn = gr.Button("🔍 Analyze Resume", variant="primary")

        # ── Right Column: Outputs ──
        with gr.Column(scale=1):
            score_out = gr.Textbox(label="🎯 Match Score", interactive=False)
            category_out = gr.Textbox(label="📂 Predicted Job Category", interactive=False, lines=5)
            matched_out = gr.Textbox(label="✅ Matched Skills", interactive=False, lines=3)
            missing_out = gr.Textbox(label="❌ Missing Skills (Skill Gap)", interactive=False, lines=3)
            verdict_out = gr.Textbox(label="💡 AI Recommendation", interactive=False, lines=3)

    chart_out = gr.Plot(label="📊 Model Accuracy Comparison")

    analyze_btn.click(
        fn=analyze_resume,
        inputs=[pdf_input, job_input, model_selector],
        outputs=[score_out, category_out, matched_out, missing_out, verdict_out, chart_out]
    )

    gr.HTML("""
    <div style="text-align:center;margin-top:2rem;color:#44445a;font-size:0.78rem;">
        Built with Scikit-learn · TF-IDF · Gradio &nbsp;|&nbsp; 5 Models Compared
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)
