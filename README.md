# 🛒 Product Category Classifier

Model de machine learning care **sugerează automat categoria unui produs pe baza titlului**,
pentru ca produsele noi dintr-un magazin online să fie clasificate rapid și fără erori manuale.

```
Titlu produs: bosch wap28390gb 8kg 1400 spin
→ Categoria sugerată: Washing Machines
```

| | |
|---|---|
| **Date** | `data/products.csv`  35.311 produse (30.824 după curățare), 10 categorii |
| **Model** | Linear SVC pe TF-IDF de caractere + TF-IDF de cuvinte + statistici ale titlului |
| **Acuratețe pe test** | **98,9%** (F1 macro 0,99) |
| **Titluri de test din enunț** | 6 / 6 corecte |

Categorii: CPUs, Digital Cameras, Dishwashers, Freezers, Fridge Freezers, Fridges, Microwaves, Mobile Phones, TVs, Washing Machines.

---

## 📁 Structura proiectului

```
product-category-classifier/
├── data/
│   └── products.csv                      # setul de date original
├── model/
│   └── product_category_model.pkl        # modelul antrenat (generat de train_model.py)
├── notebooks/
│   └── product_category_analysis.ipynb   # analiza completă: explorare, curățare, features, comparare modele
├── features.py                           # funcții comune: curățare, caracteristici, definiția modelului
├── train_model.py                        # antrenează și salvează modelul
├── predict_category.py                   # testare interactivă în consolă
├── requirements.txt
└── README.md
```

**De ce există `features.py`?** Curățarea și definiția modelului sunt folosite în notebook, la antrenare și la predicție.
Le țin într-un singur loc ca logica să fie identică peste tot. În plus, modelul salvat folosește funcția
`title_stats` din acest fișier, deci `predict_category.py` trebuie să o poată importa.

---

## 🚀 Cum rulezi proiectul

### 1. Instalare
```bash
git clone https://github.com/<utilizator>/product-category-classifier.git
cd product-category-classifier
pip install -r requirements.txt
```

### 2. Testare interactivă (modelul este deja inclus)
```bash
python predict_category.py
```
Scrii titlul produsului și apeși Enter. Pentru ieșire scrii `exit`.

Exemple de încercat:

| Titlu | Categoria așteptată |
|---|---|
| `iphone 7 32gb gold,4,3,Apple iPhone 7 32GB` | Mobile Phones |
| `olympus e m10 mark iii geh use silber` | Digital Cameras |
| `kenwood k20mss15 solo` | Microwaves |
| `bosch wap28390gb 8kg 1400 spin` | Washing Machines |
| `bosch serie 4 kgv39vl31g` | Fridge Freezers |
| `smeg sbs8004po` | Fridge Freezers |

### 3. Reantrenarea modelului (opțional)
```bash
python train_model.py
```
Scriptul curăță datele, afișează acuratețea pe 20% date de test, apoi antrenează modelul final pe toate datele
și îl salvează în `model/product_category_model.pkl`. Durează sub un minut.

> 💡 Dacă la `predict_category.py` apare o eroare la încărcarea modelului (de obicei din cauza unei versiuni diferite
> de scikit-learn), rulează `python train_model.py` ca să generezi modelul din nou pe calculatorul tău.

### 4. Notebook-ul de analiză
```bash
jupyter notebook notebooks/product_category_analysis.ipynb
```
Notebook-ul este salvat cu toate rezultatele, deci poate fi citit și direct pe GitHub, fără să fie rulat.

---

## 🔍 Pașii și deciziile principale

### Curățarea datelor
| Problemă găsită | Ce am făcut |
|---|---|
| Nume de coloane cu spații/underscore (`' Category Label'`, `'_Product Code'`) | standardizate: `category_label`, `product_code` |
| 172 titluri și 44 categorii lipsă | rânduri eliminate |
| Etichete scrise diferit: `fridge`, `CPU`, `Mobile Phone` | corectate în `Fridges`, `CPUs`, `Mobile Phones` |
| ~4.450 titluri duplicate (același produs, alți vânzători) | eliminate, ca același titlu să nu ajungă și în train și în test |

### Ce coloane folosesc
Doar **titlul**. `number_of_views` și `merchant_rating` au aceeași distribuție în toate categoriile
(nu ajută), iar un produs nou nici nu are încă vizualizări. `product_code`, `merchant_id` și `listing_date` nu au legătură cu categoria.

### Ingineria caracteristicilor
- **TF-IDF pe caractere (2–5)**: cea mai utilă idee. Prinde bucăți din codurile de model (`kgv`, `wap28`, `sbs`),
  care apar la multe produse din aceeași categorie. A adus ~+3 puncte procentuale față de cuvinte.
- **TF-IDF pe cuvinte (1–2)**: prinde cuvinte-cheie întregi (`solo`, `washing machine`).
- **Statistici ale titlului**: număr de cuvinte/caractere/cifre, cel mai lung cuvânt, caractere speciale.
  Arată diferențe clare între categorii (ex. titlurile CPU sunt mult mai lungi), dar în model au adus un câștig neglijabil.
- ❌ *Majuscule (USB, LED)*: toate titlurile sunt deja scrise cu litere mici, deci ideea nu se aplică pe aceste date.

### Compararea modelelor (setul de test, 20%)
| Vectorizare | Naive Bayes | Logistic Regression | Linear SVC |
|---|---|---|---|
| TF-IDF cuvinte | 97,4% | 95,9% | 96,4% |
| TF-IDF caractere | 98,1% | 98,9% | **99,2%** |

| Combinație (Linear SVC) | Acuratețe | Titluri din enunț |
|---|---|---|
| A. caractere | **99,2%** | 5 / 6 |
| B. caractere + cuvinte | 98,9% | 6 / 6 |
| **C. caractere + cuvinte + statistici** ✅ | 98,9% | **6 / 6** |

**De ce C și nu A?** A are cu ~0,2 pp mai multă acuratețe (confirmat și prin validare încrucișată pe 5 fold-uri),
dar a clasificat greșit `kenwood k20mss15 solo` (Fridges în loc de Microwaves). Colegii vor introduce des titluri scurte,
doar marcă + cod, iar C se descurcă mai bine pe ele. Este un compromis conștient, documentat în notebook.

### Limitări cunoscute
- Majoritatea greșelilor sunt între **Fridges, Freezers și Fridge Freezers**, produse foarte asemănătoare.
- Decizia A vs C s-a bazat pe doar 6 exemple manuale.

### Idei pentru versiunea următoare
- afișarea **top 3 categorii** cu scor, pentru cazurile în care modelul nu e sigur;
- un set de test separat cu titluri scurte (marcă + cod);
- prefixe de coduri pe producător pentru categoriile de frigidere;
- reantrenare periodică pe produsele noi, validate de colegi.

---

**Autor:** Cristiana-Raluca Costache · Tehnologii: Python, pandas, scikit-learn, matplotlib, seaborn
