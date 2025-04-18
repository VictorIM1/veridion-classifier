import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Incarcam fisierele procesate
companies_df = pd.read_csv("data/companies_clean.csv")
with open("data/taxonomy_expanded.json", "r") as f:
    taxonomy_dict = json.load(f)

# Initializam encoderul
model = SentenceTransformer("all-MiniLM-L6-v2")

# Construim lista de labeluri si descrieri
taxonomy_labels = list(taxonomy_dict.keys())
taxonomy_descriptions = [" ".join(taxonomy_dict[label]) for label in taxonomy_labels]

# Generam embeddings
taxonomy_embeddings = model.encode(taxonomy_descriptions, show_progress_bar=True)
company_embeddings = model.encode(companies_df["cleaned_text"].tolist(), show_progress_bar=True)

# Clasificam fiecare companie
predicted_labels = []
for emb in company_embeddings:
    sims = cosine_similarity([emb], taxonomy_embeddings)[0]
    best_idx = np.argmax(sims)
    predicted_labels.append(taxonomy_labels[best_idx])

# Adaugam predictiile in dataframe
companies_df["predicted_label"] = predicted_labels
companies_df.to_csv("data/classified_companies.csv", index=False)

print("Saved classified companies to data/classified_companies.csv")
