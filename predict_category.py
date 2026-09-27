
import pickle
import sys
from pathlib import Path

# Importul este necesar: modelul salvat folosește funcția title_stats din features.py
import features  # noqa: F401
from features import clean_title

MODEL_PATH = Path(__file__).resolve().parent / "model" / "product_category_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        sys.exit("Nu găsesc modelul. Rulează mai întâi:  python train_model.py")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def main():
    model = load_model()
    print("=" * 60)
    print("Clasificator de produse - scrie titlul produsului și apasă Enter.")
    print("Categorii posibile:", ", ".join(model.classes_))
    print("Pentru ieșire scrie: exit")
    print("=" * 60)

    while True:
        try:
            title = input("\nTitlu produs: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if title.lower() in {"exit", "quit", "q"}:
            break
        if not title:
            print("Nu ai scris nimic, încearcă din nou.")
            continue

        # Aceeași curățare ca la antrenare (litere mici, spații normalizate)
        category = model.predict([clean_title(title)])[0]
        print(f"Categoria sugerată: {category}")

    print("\nLa revedere!")


if __name__ == "__main__":
    main()
