# Heart Disease Prediction Using 3 ML Algorithms

A comparative study of **K-Nearest Neighbors (KNN)**, **Naive Bayes**, and **Decision Tree** for predicting heart disease using the UCI Cleveland Heart Disease dataset. Models are tuned with `GridSearchCV` and evaluated using 10-fold cross-validation.

This project reimplements the general methodology of:
> G. A. Ansari et al., *"Performance Evaluation of Machine Learning Techniques (MLT) for Heart Disease Prediction,"* Computational and Mathematical Methods in Medicine, 2023.

---

## Dataset

- **Source:** [UCI Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Records:** 303 patients
- **Features:** 13 input attributes + 1 target
- **Target:** Originally 5 classes (0–4), converted to **binary classification** — `0 = No Disease`, `1 = Disease`

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia result |

---

## Pipeline

```
UCI Heart Disease Dataset
        ↓
Data Preprocessing
        ↓
Missing Value Imputation (mode imputation on ca, thal)
        ↓
Target Conversion (num → binary target)
        ↓
Exploratory Data Analysis (EDA)
        ↓
Apply ML Technique
        ↓
   ┌─────────────┬──────────────┬───────────────┐
   │     KNN     │  Naive Bayes │ Decision Tree │
   └─────────────┴──────────────┴───────────────┘
        ↓
GridSearchCV Hyperparameter Tuning (per model)
        ↓
10-Fold Cross-Validation
        ↓
Performance Evaluation
        ↓
Final Model Comparison
```

---

## Preprocessing

- **Missing values:** `ca` (4 missing) and `thal` (2 missing) imputed using **mode**, since both are categorical.
- **Duplicates:** None found.
- **Feature scaling:** `StandardScaler` applied only for KNN (distance-based algorithm). Naive Bayes and Decision Tree use raw feature values.
- **Feature selection:** Correlation analysis performed against the target; all 13 features retained (no removal).

---

## Models & Hyperparameter Tuning

Each model is tuned using `GridSearchCV` (`cv=10`, `scoring="accuracy"`) and then evaluated using a separate outer `StratifiedKFold(n_splits=10)` loop.

| Model | Hyperparameters Tuned | Best Configuration |
|---|---|---|
| **KNN** | `n_neighbors`, `weights`, `metric` | `n_neighbors=13`, `weights=distance`, `metric=euclidean` |
| **Naive Bayes** | `var_smoothing` | `var_smoothing=1e-11` |
| **Decision Tree** | `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf` | `criterion=gini`, `max_depth=3`, `min_samples_split=5`, `min_samples_leaf=10` |

---

## Results (10-Fold Cross-Validation)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| KNN | 82.84% | 84.25% | 76.98% | 80.45% |
| **Naive Bayes** | **83.83%** | **84.62%** | **79.14%** | **81.78%** |
| Decision Tree | 79.87% | 83.05% | 70.50% | 76.26% |

**Mean CV Accuracy ± Standard Deviation**

| Model | Mean Accuracy | Std. Dev |
|---|---:|---:|
| KNN | 82.80% | ±6.16% |
| **Naive Bayes** | **83.78%** | ±6.10% |
| Decision Tree | 79.80% | ±7.92% |

**Naive Bayes achieved the best overall performance**, with the highest accuracy, precision, recall, and F1-score, along with the most consistent performance across folds (lowest standard deviation alongside KNN).

---

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

---

## Project Structure

```
├── FINAL_NOTEBOOK.ipynb      # Full implementation: preprocessing, EDA, modeling, evaluation
├── FINAL_REPORT.pdf          # Written report with methodology and results
├── processed.cleveland.csv   # Cleaned dataset used for modeling
└── README.md
```

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/Heart-Disease-Prediction-Using-3-ML-Algorithms.git
   cd Heart-Disease-Prediction-Using-3-ML-Algorithms
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. Launch the notebook
   ```bash
   jupyter notebook FINAL_NOTEBOOK.ipynb
   ```

---

## Author

**Ujjwal Dahal**
Department of Electronics and Computer Engineering, IOE Thapathali Campus
📧 ujjwaldahal20@gmail.com

---

## Acknowledgment

This project reimplements the general workflow presented in Ansari et al. (2023). Results differ from the original paper due to differences in preprocessing, feature handling, and experimental setup — the goal was to reproduce the methodology, not the exact reported numbers.
