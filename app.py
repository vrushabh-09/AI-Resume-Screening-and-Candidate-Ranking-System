import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import extract_text, extract_skills, extract_experience, compute_similarity, rank_resumes

# Streamlit Page Config
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("📄 AI Resume Screening System")

# Upload Job Description
st.subheader("📝 Upload Job Description")
job_desc = st.text_area("Paste the job description here")

# Upload Resumes
st.subheader("📂 Upload Resumes (PDF/DOCX)")
uploaded_files = st.file_uploader(
    "Upload multiple resumes", accept_multiple_files=True, type=["pdf", "docx"])

if uploaded_files and job_desc:
    candidates = []

    with st.spinner("Processing resumes... ⏳"):
        for file in uploaded_files:
            resume_text = extract_text(file)
            skills = extract_skills(resume_text)
            # Fixed experience extraction
            experience = extract_experience(resume_text)
            similarity = compute_similarity(job_desc, resume_text)

            candidates.append({
                "Candidate": file.name,
                "Skills": ", ".join(skills),
                "Experience": experience,
                "Similarity Score": similarity
            })

        ranked_candidates = rank_resumes(candidates)
        df = pd.DataFrame(ranked_candidates)

    # Display Results
    st.subheader("📊 Ranked Candidates")
    st.write(df)

    # Visualization
    st.subheader("📈 Similarity Scores")
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Similarity Score", y="Candidate", hue="Candidate",
                data=df, palette="Blues_r", legend=False)
    plt.xlabel("Similarity Score (%)")
    plt.ylabel("Candidate")
    st.pyplot(plt)
