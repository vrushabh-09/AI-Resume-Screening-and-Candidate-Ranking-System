import subprocess
import sys

dependencies = [
    "streamlit", "numpy", "pandas", "matplotlib", "seaborn",
    "spacy", "scikit-learn", "PyMuPDF", "python-docx"
]

for package in dependencies:
    subprocess.run([sys.executable, "-m", "pip", "install", package])

# Download SpaCy model
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

print("✅ All dependencies installed successfully!")
