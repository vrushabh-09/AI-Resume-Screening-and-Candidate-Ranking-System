import fitz  # PyMuPDF for PDF processing
import docx
import re
import spacy
import numpy as np
from spacy.matcher import PhraseMatcher
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy",
                   "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Predefined Skill Set
SKILL_SET = {
    "python", "java", "c++", "html", "css", "javascript", "sql",
    "machine learning", "deep learning", "flask", "django",
    "tensorflow", "pytorch", "nlp", "data science", "cloud computing",
    "aws", "react", "angular", "node.js", "docker", "kubernetes",
    "git", "linux"
}


def extract_text(file):
    """Extract text from PDF or DOCX resumes."""
    try:
        if file.name.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return " ".join([page.get_text("text") for page in doc])
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            return " ".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        return f"❌ Error processing {file.name}: {e}"
    return ""


def extract_skills(text):
    """Extract skills using NLP PhraseMatcher with improved normalization."""
    matcher = PhraseMatcher(
        nlp.vocab, attr="LEMMA")  # Use lemmatization to improve matching
    # Convert all skills to lowercase
    skill_patterns = [nlp(skill.lower()) for skill in SKILL_SET]
    matcher.add("SKILLS", skill_patterns)

    doc = nlp(text.lower())  # Normalize text
    matches = matcher(doc)

    extracted_skills = {doc[start:end].text.strip()
                        for _, start, end in matches}

    return list(extracted_skills)


def extract_experience(text):
    """
    Extract years of experience from text with improved accuracy.
    Handles cases like:
      - '5 years of experience' → 5
      - '2 to 4 years' → 4
      - '10+ years' → 10
      - 'More than 3.5 years' → 3.5
      - '7+ years' → 7
    Avoids extracting unrelated numbers like phone numbers.
    """
    exp_regex = r"(?:(?:more than|over|at least|around|about|approximately)?\s*(\d{1,2}(\.\d{1,2})?)\s*(?:\+|-|to)?\s*(\d{1,2}(\.\d{1,2})?)?\s*(?:year|years|yr|yrs))"

    matches = re.findall(exp_regex, text.lower())

    years = []
    for match in matches:
        try:
            if match[0]:  # First number (X years)
                years.append(float(match[0]))
            if match[2]:  # Second number (Y in 'X to Y years')
                years.append(float(match[2]))
        except ValueError:
            continue

    # Ensure we are extracting realistic experience values (1-50 years)
    valid_years = [y for y in years if 1 <= y <= 50]

    return round(max(valid_years, default=0), 1)


def compute_similarity(job_desc, resume_text):
    """Compute similarity using word embeddings."""
    job_desc_doc = nlp(job_desc)
    resume_doc = nlp(resume_text)

    if job_desc_doc.vector_norm == 0 or resume_doc.vector_norm == 0:
        return 0  # Avoids zero-vector error

    similarity = cosine_similarity([job_desc_doc.vector], [
                                   resume_doc.vector])[0][0]
    return round(similarity * 100, 2)


def rank_resumes(resumes):
    """Rank resumes based on similarity and experience."""
    return sorted(resumes, key=lambda x: (x["Similarity Score"] * 0.7 + x["Experience"] * 0.3), reverse=True)
