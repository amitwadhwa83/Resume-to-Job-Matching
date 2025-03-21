import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Load SBERT Model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def scrape_job_data():
    # TODO: Implement web scraping for real-time job postings.
    # Return data in a Pandas DataFrame with columns: job_title, description, skills, date_posted, location.
    return pd.DataFrame(columns=["title", "description", "skills_desc", "original_listed_time", "location"])

# Load & Preprocess Data
def load_data(file_path):
    """Loads job dataset and filters for Europe-based jobs."""
    if file_path:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["title", "description", "skills_desc"])
        #df = df[df["location"].str.contains("Europe", na=False)]
    else:
        df = scrape_job_data()  # Hook for future real-time scraping
        # df = df[df["location"].str.contains("Europe", na=False)]
    return df

# Generate Embeddings
def generate_embeddings(texts):
    """Generates SBERT embeddings for a list of texts."""
    return np.array([model.encode(text) for text in texts])

# Train KNN Model
def  train_knn(embeddings, n_neighbors=5):
    """Trains a KNN model for similarity search."""
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(embeddings)
    return knn

# Save Model & Data
def save_model(data, embeddings, knn, path="models/"):
    """Saves the trained KNN model and embeddings."""
    with open(f"{path}knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)
    np.save(f"{path}embeddings.npy", embeddings)
    data.to_csv(f"{path}job_data.csv", index=False)

if __name__ == "__main__":
    # Process dataset and train model
    file_path = "data/postings.csv"  # Ensure this file exists
    print("Loading data")
    data = load_data(file_path)
    print(f"generating embedding for data size : {len(data)}")
    embeddings = generate_embeddings(data["description"])
    print("Training model for embedding")
    knn = train_knn(embeddings)
    print("Saving model")
    save_model(data, embeddings, knn)
    print("Model training completed and saved!")
