
import pickle
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from features import build_model, load_and_clean

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "products.csv"
MODEL_PATH = BASE_DIR / "model" / "product_category_model.pkl"


def main():
    # 1. Date curate
    df = load_and_clean(DATA_PATH)
    X, y = df["product_title"], df["category_label"]
    print(f"Produse după curățare: {len(df)} | categorii: {y.nunique()}")

    # 2. Verificare pe date nevăzute (aceeași împărțire ca în notebook: 80/20, stratify, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    check_model = build_model().fit(X_train, y_train)
    y_pred = check_model.predict(X_test)
    print(f"\nAcuratețe pe setul de test: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    # 3. Modelul final pe toate datele
    final_model = build_model().fit(X, y)

    # 4. Salvare
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Model salvat în {MODEL_PATH.relative_to(BASE_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
