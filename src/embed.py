import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_data(data_dir: str):
    companies_path = os.path.join(data_dir, 'companies_clean.csv')
    taxonomy_path = os.path.join(data_dir, 'taxonomy_expanded.json')
    df = pd.read_csv(companies_path)
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    return df, taxonomy

def generate_embeddings(texts: list, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Generate embeddings for a list of texts using SentenceTransformer.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def main():
    # Paths
    data_dir = os.path.join(os.getcwd(), 'data')
    out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load cleaned data and expanded taxonomy
    df, taxonomy = load_data(data_dir)

    # Company embeddings
    company_texts = df['cleaned_text'].astype(str).tolist()
    company_embeddings = generate_embeddings(company_texts)
    np.save(os.path.join(out_dir, 'company_embeddings.npy'), company_embeddings)
    print(f"Saved company_embeddings.npy with shape {company_embeddings.shape}")

    # Label embeddings: use the variants or just the label itself
    labels = list(taxonomy.keys())
    # We can embed the label names directly
    label_embeddings = generate_embeddings(labels)
    np.save(os.path.join(out_dir, 'label_embeddings.npy'), label_embeddings)
    print(f"Saved label_embeddings.npy with shape {label_embeddings.shape}")

if __name__ == '__main__':
    main()
