import os
import csv
import re
import fitz  # PyMuPDF
import nltk
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv

# ========== Initial Setup ==========
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load Gemini API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-8b")

# ========== Helper Functions ==========
def extract_text_from_pdf(file):
    try:
        text = ""
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text("text")
        return clean_text(text)
    except Exception as e:
        return f"Error reading PDF: {e}"

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered)

def get_embedding(text):
    return model_embed.encode(text, convert_to_tensor=True)

def cosine_similarity_score(resume_emb, jd_emb):
    return float(util.pytorch_cos_sim(resume_emb, jd_emb)[0][0])

def generate_gemini_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def store_history(resume, jd, score, skills, roles):
    row = [resume[:50], jd[:50], score, skills, roles]
    try:
        with open("history.csv", "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(row)
    except Exception as e:
        print(f"Error saving history: {e}")

# ========== Streamlit App ==========
st.set_page_config(page_title="AI Resume Matcher & Advisor", layout="wide")
st.title("🤖 AI Resume Matcher & Job Advisor")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF/Text)", type=["pdf", "txt"])
with col2:
    jd_file = st.file_uploader("📝 Upload Job Description (PDF/Text)", type=["pdf", "txt"])

if st.button("🔍 Analyze") and resume_file and jd_file:
    try:
        # ✅ Extract Text
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else clean_text(resume_file.read().decode("utf-8"))
        jd_text = extract_text_from_pdf(jd_file) if jd_file.name.endswith(".pdf") else clean_text(jd_file.read().decode("utf-8"))

        # ✅ Preprocess & Embed
        resume_proc = preprocess_text(resume_text)
        jd_proc = preprocess_text(jd_text)
        resume_emb = get_embedding(resume_proc)
        jd_emb = get_embedding(jd_proc)

        # ✅ Similarity Score
        similarity = cosine_similarity_score(resume_emb, jd_emb)
        st.subheader("🔎 Similarity Score")
        st.success(f"**{similarity:.2f}** (Higher means better match!)")

        # ✅ Gemini Suggestions
        skill_prompt = f"Given this job description:\n{jd_text}\nand this resume:\n{resume_text}\n\nSuggest 5 missing skills."
        role_prompt = f"Based on this resume:\n{resume_text}\nSuggest 2–3 suitable job roles."

        skills = generate_gemini_response(skill_prompt)
        roles = generate_gemini_response(role_prompt)

        st.subheader("🧠 Suggested Skills to Improve")
        st.write(skills)

        st.subheader("💼 Recommended Job Roles")
        st.write(roles)

        # ✅ Store History
        store_history(resume_text, jd_text, similarity, skills, roles)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

# ✅ Display Upload History
st.markdown("---")
st.subheader("📊 Upload History")
try:
    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv", header=None)
        df.columns = ["Resume Snippet", "Job Description Snippet", "Similarity", "Suggested Skills", "Recommended Roles"]
        st.dataframe(df)
    else:
        st.info("No history found yet. Upload resumes and job descriptions to begin!")
except Exception as e:
    st.error(f"Error reading history: {e}")
