## **`Disaster Tweet Classifier`**

A complete end-to-end NLP project that classifies tweets as **disaster-related** or **non-disaster** using machine learning. The pipeline covers data exploration, feature engineering, multi-model training, evaluation, and a live Flask web application for real-time predictions.

---

## **Project Structure**

```
├── 1_Data_Exploration_Preparation.ipynb       # Step 1 — EDA, text cleaning, train/test split
├── 2_Feature_Engineering_Model_Training.ipynb # Step 2 — TF-IDF features + model training
├── 3_Model_Evaluation.ipynb                   # Step 3 — Metrics, curves, error analysis
├── 4_Deployment_Web_App.ipynb                 # Step 4 — Flask app generation
│
├── artifacts/                                 # Intermediate data files (auto-generated)
│   ├── train_prepared.csv
│   └── test_prepared.csv
│
├── model_store/                               # Saved model files (auto-generated)
│   ├── best_disaster_tweet_model.joblib
│   └── model_comparison.csv
│
├── reports/                                   # Evaluation outputs (auto-generated)
│   ├── classification_report.csv
│   └── misclassified_examples.csv
│
├── web_app/
│   ├── app.py                                 # Flask application
│   └── templates/
│       └── index.html                         # Prediction UI
│
└── README.md
```

---

## **Pipeline Overview**

Run the notebooks **in order**. Each notebook produces files consumed by the next.

```
twitter_disaster.csv
  └─▶ [1] Data Exploration & Preparation
          └─▶ artifacts/train_prepared.csv
          └─▶ artifacts/test_prepared.csv
                  └─▶ [2] Feature Engineering & Model Training
                          └─▶ model_store/best_disaster_tweet_model.joblib
                          └─▶ model_store/model_comparison.csv
                                  └─▶ [3] Model Evaluation
                                          └─▶ reports/classification_report.csv
                                          └─▶ reports/misclassified_examples.csv
                                                  └─▶ [4] Deployment → web_app/app.py
                                                                └─▶ Flask Web App
```

---

## **Notebook Descriptions**

### Notebook 1 — Data Exploration & Preparation
**File:** `1_Data_Exploration_Preparation.ipynb`

Loads the raw disaster tweet dataset and prepares it for modelling.

- Inspects dataset shape, column types, and missing values
- Plots **target class distribution** (disaster vs. non-disaster)
- Engineers EDA features: `text_length_chars`, `word_count`, `has_hashtag`, `has_mention`, `has_url`
- Compares feature means across both classes and visualizes text-length distributions
- Analyzes the top 20 tweet keywords
- Applies a reproducible **text cleaning** function (lowercasing, URL/HTML/mention removal, punctuation stripping)
- Performs an 80/20 **stratified train/test split**
- **Output:** `artifacts/train_prepared.csv`, `artifacts/test_prepared.csv`

---

## **Notebook 2 — Feature Engineering & Model Training**
**File:** `2_Feature_Engineering_Model_Training.ipynb`

Builds a combined text + numeric feature pipeline and benchmarks four classifiers.

## **Feature design:**

| Feature | Type | Description |
|---|---|---|
| `clean_text` | Text | TF-IDF (unigrams + bigrams, 20,000 features, sublinear TF) |
| `text_length_chars` | Numeric | Character count of the original tweet |
| `word_count` | Numeric | Word count after cleaning |
| `has_hashtag` | Binary | Whether the tweet contains `#` |
| `has_mention` | Binary | Whether the tweet contains `@` |
| `has_url` | Binary | Whether the tweet contains a URL |

## **Models trained:**

| Model | Notes |
|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000` |
| Linear SVC | `class_weight="balanced"` |
| Multinomial Naive Bayes | Text-only baseline |
| Random Forest | 300 trees, `class_weight="balanced_subsample"` |

- Evaluates each model with **5-fold cross-validation F1** and hold-out test metrics (accuracy, precision, recall, F1)
- Selects and retrains the best model on the full training set
- **Output:** `model_store/best_disaster_tweet_model.joblib`, `model_store/model_comparison.csv`

---

## **Notebook 3 — Model Evaluation**
**File:** `3_Model_Evaluation.ipynb`

Deep evaluation of the saved best model on the held-out test set.

- Prints accuracy, precision, recall, and F1 score
- Generates a full **classification report** (per-class and macro averages)
- Plots a **confusion matrix** with labeled axes
- Plots the **ROC curve** with AUC score (handles both `predict_proba` and `decision_function` models)
- Plots the **Precision-Recall curve**
- Performs **error analysis** — samples 20 misclassified tweets to identify failure patterns (sarcasm, ambiguity, metaphor, slang)
- **Output:** `reports/classification_report.csv`, `reports/misclassified_examples.csv`

> **Interpretation guide:** High precision = fewer false alarms. High recall = fewer missed disasters. F1 balances both. Misclassified examples surface edge cases for future improvement.

---

## **Notebook 4 — Deployment (Web App)**
**File:** `4_Deployment_Web_App.ipynb`

Generates the complete Flask web application for real-time tweet classification.

- Loads the saved model pipeline
- Defines the same `clean_text` and `build_features` preprocessing functions used during training (ensuring consistency between training and inference)
- Writes `web_app/app.py` and `web_app/templates/index.html`
- Returns confidence scores via `predict_proba` (or sigmoid-transformed `decision_function` for SVM)
- Shows how to run the app locally

---

## **Flask Web App (`app.py`)**

The deployed app accepts a raw tweet via a text form and returns:

- **Label:** `Disaster Tweet` or `Non-Disaster Tweet`
- **Confidence score:** probability of the predicted class

## **Running locally**

```bash
# 1. Install dependencies
pip install flask joblib pandas scikit-learn

# 2. Ensure the model exists at model_store/best_disaster_tweet_model.joblib
#    (run notebooks 1 and 2 first)

# 3. Launch the app
python web_app/app.py
```

Open your browser at `http://127.0.0.1:5000`.

---

## **Setup & Installation**

## **1. Clone the repository**

```bash
git clone https://github.com/your-username/disaster-tweet-classifier.git
cd disaster-tweet-classifier
```

## **2. Install dependencies**

```bash
pip install pandas numpy matplotlib scikit-learn flask joblib
```

## **3. Add the raw dataset**

Place the source CSV in the project root:

```
twitter_disaster.csv
```

Expected columns: `id`, `keyword`, `location`, `text`, `target` (1 = disaster, 0 = non-disaster)

## **4. Run the notebooks in order**

```
1 → 2 → 3 → 4
```

## **5. Launch the web app**

```bash
python web_app/app.py
```

---

## **Dataset**

| Attribute | Value |
|---|---|
| Source | Twitter (Kaggle NLP Disaster Tweets) |
| Labels | 1 = Disaster, 0 = Non-Disaster |
| Key columns | `text`, `keyword`, `location`, `target` |
| Split | 80% train / 20% test (stratified) |

---

## **Tech Stack**

- **Data processing:** pandas, NumPy
- **NLP / Features:** scikit-learn (TF-IDF, ColumnTransformer, Pipeline)
- **Models:** Logistic Regression, Linear SVC, Multinomial Naive Bayes, Random Forest
- **Evaluation:** scikit-learn metrics (ROC-AUC, PR curve, confusion matrix)
- **Serialization:** joblib
- **Web app:** Flask

---

## **Key Design Decisions**

- The **same `clean_text` function** is used in both training (Notebook 1) and the Flask app, preventing training-serving skew.
- **Numeric metadata features** (tweet length, hashtag/mention/URL presence) are combined with TF-IDF text features inside a single `ColumnTransformer` pipeline, so the entire preprocessing + model is saved as one `.joblib` file — no separate vectorizer files needed.
- Model selection is **data-driven** — all four classifiers are scored on cross-validation F1 and held-out test F1, and the winner is chosen automatically.
- Confidence scores gracefully fall back to a sigmoid of `decision_function` for models like LinearSVC that do not expose `predict_proba`.




---

