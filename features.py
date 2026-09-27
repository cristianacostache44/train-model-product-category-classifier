
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVC

# Etichetele scrise diferit în setul de date -> forma corectă (am găsit-o în notebook, la explorare)
LABEL_FIXES = {
    "fridge": "Fridges",
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones",
}


def clean_columns(df):
    """Numele coloanelor vin cu spații și underscore-uri (ex. ' Category Label', '_Product Code').
    Le aduc la forma: category_label, product_code etc."""
    df = df.copy()
    df.columns = (df.columns.str.strip()
                            .str.lower()
                            .str.replace(" ", "_")
                            .str.lstrip("_"))
    return df


def clean_title(title):
    """Normalizez titlul: litere mici și un singur spațiu între cuvinte."""
    return re.sub(r"\s+", " ", str(title)).strip().lower()


def load_and_clean(path):
    """Încarcă products.csv și returnează un DataFrame curat, gata de antrenare."""
    df = clean_columns(pd.read_csv(path))

    # 1. Fără titlu sau fără categorie nu putem învăța nimic din rând
    df = df.dropna(subset=["product_title", "category_label"])

    # 2. Categorii: fără spații, cu variantele greșite corectate
    df["category_label"] = df["category_label"].str.strip().replace(LABEL_FIXES)

    # 3. Titluri normalizate și duplicate eliminate (același titlu de la mai mulți vânzători)
    df["product_title"] = df["product_title"].map(clean_title)
    df = df[df["product_title"] != ""]
    df = df.drop_duplicates(subset="product_title").reset_index(drop=True)
    return df


def title_stats(titles):
    """Caracteristici numerice „mici” calculate din titlu (feature engineering).

    - n_words      : numărul de cuvinte (procesoarele au titluri lungi, cu multe specificații)
    - n_chars      : lungimea titlului
    - n_digits     : câte cifre are (coduri de model, GB, kg...)
    - has_number   : 1 dacă titlul conține măcar o cifră
    - longest_word : lungimea celui mai lung cuvânt (codurile de model sunt lungi, ex. 'kgv39vl31g')
    - n_special    : caractere speciale (/, -, +, ...) – frecvente la telefoane și camere
    """
    s = pd.Series(list(titles), dtype="object").astype(str)
    words = s.str.split()
    return pd.DataFrame({
        "n_words": words.str.len(),
        "n_chars": s.str.len(),
        "n_digits": s.str.count(r"\d"),
        "has_number": s.str.contains(r"\d").astype(int),
        "longest_word": words.apply(lambda w: max(map(len, w)) if w else 0),
        "n_special": s.str.count(r"[^a-z0-9\s]"),
    })


def build_model():
    """Modelul final ales în notebook:
    TF-IDF pe caractere + TF-IDF pe cuvinte + caracteristici numerice → LinearSVC.

    - caractere (char_wb 2–5): prind bucăți din codurile de model ('kgv', 'wap28', 'sbs') și greșeli de scriere;
    - cuvinte (1–2): prind cuvinte întregi și perechi ('washing machine', 'fridge freezer');
    - caracteristici numerice: scalate la [0, 1] ca să nu domine restul.
    """
    features = FeatureUnion([
        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("title_stats", Pipeline([
            ("extract", FunctionTransformer(title_stats)),
            ("scale", MinMaxScaler()),
        ])),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LinearSVC(random_state=42)),
    ])
