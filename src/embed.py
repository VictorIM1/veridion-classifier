import pandas as pd
import re
import json
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data (for first run)
nltk.download('punkt')
nltk.download('wordnet')


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
     - lowercasing
     - remove non-alphanumeric characters
     - collapse multiple spaces
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_taxonomy(labels: list) -> dict:
    """
    Build an expanded mapping for each label:
     - original label
     - lowercase label
     - lemmatized form
    Returns a dict: {label: [variants...]}
    """
    lemmatizer = WordNetLemmatizer()
    expanded = {}
    for label in labels:
        base = label.strip()
        lower = base.lower()
        tokens = lower.split()
        lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
        lemma_label = " ".join(lemmas)
        variants = list({base, lower, lemma_label})
        expanded[base] = variants
    return expanded


def main():
    data_dir = os.path.join(os.getcwd(), 'data')
    company_file = os.path.join(data_dir, 'company_list.csv')
    taxonomy_file = os.path.join(data_dir, 'insurance_taxonomy.csv')

    out_companies = os.path.join(data_dir, 'companies_clean.csv')
    out_taxonomy = os.path.join(data_dir, 'taxonomy_expanded.json')

    # Load company data with fallback encoding
    try:
        df = pd.read_csv(company_file, encoding='latin-1', engine='python')
    except Exception:
        df = pd.read_csv(company_file, encoding='utf-8', engine='python')

    # Adjust these column names to your exact headers
    text_cols = ['description', 'business_tags', 'sector', 'category', 'niche']
    df['cleaned_text'] = df[text_cols].fillna('').agg(' '.join, axis=1).apply(clean_text)
    df.to_csv(out_companies, index=False)
    print(f"Saved cleaned companies to {out_companies}")

    # Load taxonomy labels with same encoding strategy
    try:
        tax_df = pd.read_csv(taxonomy_file, encoding='latin-1', engine='python')
    except Exception:
        tax_df = pd.read_csv(taxonomy_file, encoding='utf-8', engine='python')

    if 'label' in tax_df.columns:
        labels = tax_df['label'].dropna().astype(str).tolist()
    else:
        labels = tax_df.iloc[:, 0].dropna().astype(str).tolist()

    expanded = expand_taxonomy(labels)
    with open(out_taxonomy, 'w', encoding='utf-8') as f:
        json.dump(expanded, f, ensure_ascii=False, indent=2)
    print(f"Saved expanded taxonomy to {out_taxonomy}")


if __name__ == '__main__':
    main()
