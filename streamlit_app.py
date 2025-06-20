import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def compute_similarity(resume_text, job_desc):
    embeddings = model.encode([resume_text, job_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return round(similarity[0][0] * 100, 2)

st.set_page_config(page_title="Resume Score Matcher", layout="centered")
st.title("📄 Resume Score Matcher")
st.write("Upload your resume (PDF) and paste a job description to see how well they match.")

uploaded_file = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
job_desc = st.text_area("Paste the Job Description Here", height=200)

if uploaded_file and job_desc.strip():
    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        score = compute_similarity(resume_text, job_desc)
    st.success(f"✅ Match Score: {score}%")
