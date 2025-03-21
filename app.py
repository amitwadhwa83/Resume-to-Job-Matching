import streamlit as st
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
from io import StringIO
import pdfplumber
import docx

import warnings
warnings.filterwarnings('ignore')

# Load SBERT Model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Load Model & Data
def load_model(path="models/"):
    with open(f"{path}knn_model.pkl", "rb") as f:
        knn = pickle.load(f)
    embeddings = np.load(f"{path}embeddings.npy")
    data = pd.read_csv(f"{path}job_data.csv")
    print("loaded model")
    return data, embeddings, knn

def cluster_skills(skills, n_clusters=10):
    """Clusters skills into meaningful groups using KMeans."""
    vectorized_skills = np.array([model.encode(skill) for skill in skills])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectorized_skills)
    return dict(zip(skills, labels))

def gap_analysis(user_skills, job_skills):
    """Identifies missing skills by comparing user skills with job requirements."""
    user_skills_set = set(user_skills)
    job_skills_set = set(job_skills)
    missing_skills = job_skills_set - user_skills_set
    return list(missing_skills)

def find_matching_jobs(user_skills, data, embeddings, knn, top_n=5):
    """Finds the best matching jobs based on user skills."""
    user_embedding = model.encode(", ".join(user_skills)).reshape(1, -1)
    distances, indices = knn.kneighbors(user_embedding, n_neighbors=top_n)
    matching_jobs = data.iloc[indices[0]]
    return matching_jobs

def extract_text_from_file(uploaded_file):
    """Extracts text from uploaded PDF, DOCX, or TXT files."""
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return " ".join([para.text for para in doc.paragraphs])
    return ""


def show_skill_trends(data):
    """Visualizes demand trends for specific skills over time."""
    #data['original_listed_time'] = pd.to_datetime(data['original_listed_time'], errors='coerce')
    data['original_listed_time'] = pd.to_datetime(data['original_listed_time'], unit='ms')
    all_skills = [skill for sublist in data['skills_desc'].str.split(', ') for skill in sublist]
    skill_counts = Counter(all_skills)
    top_skills = pd.DataFrame(skill_counts.items(), columns=['Skill', 'Count']).nlargest(15, 'Count')

    st.subheader("Top In-Demand Skills")
    st.bar_chart(top_skills.set_index("Skill"))

    data['skills_desc'] = data['skills_desc'].apply(lambda x: x.split(", ") if isinstance(x, str) else [])
    skill_trends = data.explode('skills_desc').groupby(['original_listed_time', 'skills_desc']).size().reset_index(name='count')
    top_trending_skills = skill_trends[skill_trends['skills_desc'].isin(top_skills['Skill'])]

    if top_trending_skills.empty:
        st.warning("No skill trends found. Try using a different dataset.")
        return

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=top_trending_skills, x='original_listed_time', y='count', hue='skills_desc')
    plt.xticks(rotation=45)
    plt.title("Skill Demand Trends Over Time")
    plt.xlabel("Job Posting Date")
    st.pyplot(plt)

# Streamlit UI
st.title("Job Skill Matching System")
st.write("Upload your resume or manually enter skills to analyze skill gaps and job relationships.")

data, embeddings, knn = load_model()

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT format)", type=["pdf", "docx", "txt"])
user_skills = []
if uploaded_file is not None:
    extracted_text = extract_text_from_file(uploaded_file)
    user_skills = extracted_text.split(",")

# Manual Skill Entry
manual_input = st.text_area("Or Enter Your Skills (comma-separated)")
if manual_input:
    user_skills.extend(manual_input.split(","))

# Select Job Role
target_job = st.selectbox("Select a Job Role for finding missing Skills", data["title"].unique())
job_row = data[data["title"] == target_job].iloc[0]
job_skills = job_row["skills_desc"].split(", ")

if st.button("Find Matching Jobs"):
    matching_jobs = find_matching_jobs(user_skills, data, embeddings, knn)
    st.write("### Best Matching Jobs:")
    st.dataframe(matching_jobs[["title", "company_name", "location", "skills_desc","job_posting_url"]])

if st.button("Analyze Skill Gap"):
    missing_skills = gap_analysis(user_skills, job_skills)
    st.write("### Missing Skills:", missing_skills if missing_skills else "None")

if st.button("Show Skill Demand Trends"):
    show_skill_trends(data)



