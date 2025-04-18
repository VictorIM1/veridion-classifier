import pandas as pd
import re
import json
import os
import chardet
import io
import csv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('wordnet')


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_taxonomy(labels: list) -> dict:
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


def detect_encoding(filepath: str, n_bytes: int = 100_000) -> str:
    with open(filepath, 'rb') as f:
        raw = f.read(n_bytes)
    result = chardet.detect(raw)
    return result['encoding'] or 'utf-8'


def detect_delimiter(text: str) -> str:
    """
    Încearcă să detecteze delimitatorul (tab, virgulă, punct și virgulă, pipe).
    Dacă Sniffer nu reușește, folosește ',' implicit.
    """
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=[',', '\t', ';', '|'])
        return dialect.delimiter
    except csv.Error:
        print("Could not detect delimiter, defaulting to comma")
        return ','



def load_csv_with_detect(path: str) -> pd.DataFrame:
    # detect encoding
    enc = detect_encoding(path)
    print(f"Detected encoding for {os.path.basename(path)}: {enc}")
    # citește eșantion pentru delimitator
    with open(path, 'r', encoding=enc, errors='replace') as f:
        sample = f.read(2048)   # primele ~2KB
    delim = detect_delimiter(sample)
    print(f"Detected delimiter for {os.path.basename(path)}: {repr(delim)}")
    # citește întreg fișierul
    with open(path, 'r', encoding=enc, errors='replace') as f:
        text = f.read()
    return pd.read_csv(io.StringIO(text), sep=delim, engine='python')


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    company_file = os.path.join(data_dir, 'company_list.csv')
    taxonomy_file = os.path.join(data_dir, 'insurance_taxonomy.csv')

    out_companies = os.path.join(data_dir, 'companies_clean.csv')
    out_taxonomy = os.path.join(data_dir, 'taxonomy_expanded.json')

    # Încarcă și curăță datele de companie
    df = load_csv_with_detect(company_file)
    text_cols = ['description', 'business_tags', 'sector', 'category', 'niche']
    df['cleaned_text'] = (
        df[text_cols]
        .fillna('')
        .agg(' '.join, axis=1)
        .apply(clean_text)
    )
    df.to_csv(out_companies, index=False)
    print(f"Saved cleaned companies to {out_companies}")

    # Încarcă și extinde taxonomia
    tax_df = load_csv_with_detect(taxonomy_file)
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
