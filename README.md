# Heart Disease Prediction Using 3 ML Algorithms

A machine learning project that predicts the presence of heart disease using patient health data. Three classification algorithms, **K-Nearest Neighbors (KNN)**, **Naive Bayes**, and **Decision Tree**, are trained, tuned, and compared on the same dataset under the same conditions to see which one performs best.

This project reimplements the general methodology of:
> G. A. Ansari et al., *"Performance Evaluation of Machine Learning Techniques (MLT) for Heart Disease Prediction,"* Computational and Mathematical Methods in Medicine, 2023.

**Note:** The original paper's methodology did not include hyperparameter tuning. In this project, **GridSearchCV-based hyperparameter tuning** was added as an extension to the original workflow, to find the best configuration for each model before final evaluation. This is an enhancement beyond what the original study did, not a straight reimplementation of it.

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide, and early prediction can help doctors and patients act before it's too late. This project explores whether simple, well-understood machine learning models can reliably classify whether a patient has heart disease based on basic clinical measurements (age, blood pressure, cholesterol, ECG results, etc.).

**Why these three algorithms?** They represent three very different approaches to classification:
- **KNN**: distance-based, makes decisions by looking at nearby data points
- **Naive Bayes**: probability-based, uses Bayes' theorem
- **Decision Tree**: rule-based, splits data using a series of if/else decisions

Comparing them side by side shows how differently these approaches behave on the same real-world medical data.

## 📊 Dataset Information

- **Name:** UCI Heart Disease Dataset (Cleveland subset)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Total records:** 303 patients
- **Total features:** 13 input features + 1 target column
- **Missing values:** 6 total (`ca`: 4 missing, `thal`: 2 missing), handled with mode imputation
- **Duplicate records:** None

### Features Used

| # | Feature | Description |
|---|---|---|
| 1 | age | Age of the patient (years) |
| 2 | sex | Gender (1 = male, 0 = female) |
| 3 | cp | Chest pain type |
| 4 | trestbps | Resting blood pressure (mm Hg) |
| 5 | chol | Serum cholesterol (mg/dl) |
| 6 | fbs | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| 7 | restecg | Resting electrocardiogram results |
| 8 | thalach | Maximum heart rate achieved |
| 9 | exang | Exercise-induced angina (1 = yes, 0 = no) |
| 10 | oldpeak | ST depression induced by exercise |
| 11 | slope | Slope of the peak exercise ST segment |
| 12 | ca | Number of major vessels colored by fluoroscopy |
| 13 | thal | Thalassemia test result |

### Target Variable

The original dataset's `num` column had 5 classes (0-4), where 0 meant no disease and 1-4 represented increasing severity. For this project, it was converted into a **binary classification problem**:

```
target = 0   ->  No heart disease   (original num = 0)
target = 1   ->  Heart disease present   (original num = 1, 2, 3, or 4)
```

## ⚙️ Implementation Details

### 1. Data Preprocessing
- Loaded the raw Cleveland dataset and assigned proper column names (no headers in the original file).
- Checked for missing values and duplicates.
- Filled missing values in `ca` and `thal` using **mode imputation**, since both are categorical features (mode of `ca` = 0, mode of `thal` = 3).
- Converted the multi-class `num` column into a binary `target` column.
- Split features into **numerical** (age, trestbps, chol, thalach, oldpeak) and **categorical** (sex, cp, fbs, restecg, exang, slope, ca, thal) groups for analysis.

### 2. Exploratory Data Analysis (EDA)
- Plotted distributions of all numerical and categorical features.
- Built a correlation matrix to see how features relate to each other and to the target.
- Ranked features by their correlation with the target. `thal`, `ca`, `exang`, `oldpeak`, and `cp` showed the strongest relationships, while `fbs` and `chol` showed the weakest.
- All 13 features were kept for modeling (no features were dropped), since even weakly-correlated features can still contribute to a combined model.

### 3. Model Training & Hyperparameter Tuning
Each model went through the same process:
1. **GridSearchCV** (with internal 10-fold CV) searched for the best hyperparameters.
2. The best hyperparameters were then used to build the final version of the model.
3. The tuned model was evaluated using an **outer 10-fold Stratified Cross-Validation** loop. The same folds were reused across all three models for a fair, apples-to-apples comparison.

**KNN** needed one extra step: since it relies on distance calculations, feature scaling (`StandardScaler`) was applied. This scaling was done *inside* the GridSearchCV pipeline itself, so the hyperparameter search wouldn't be biased by unscaled features (e.g., `chol` being on a much bigger scale than `oldpeak`). Naive Bayes and Decision Tree don't need scaling, so raw feature values were used for those.

### 4. Final Hyperparameters Selected

| Model | Tuned Parameters | Best Values Found |
|---|---|---|
| **KNN** | `n_neighbors`, `weights`, `metric` | `n_neighbors=13`, `weights=distance`, `metric=euclidean` |
| **Naive Bayes** | `var_smoothing` | `var_smoothing=1e-11` |
| **Decision Tree** | `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf` | `criterion=gini`, `max_depth=3`, `min_samples_split=5`, `min_samples_leaf=10` |

### 5. Evaluation
Each model's final performance was measured using:
- **Accuracy**: overall percentage of correct predictions
- **Precision**: of the patients predicted to have disease, how many actually did
- **Recall**: of the patients who actually had disease, how many were correctly caught
- **F1-Score**: balance between precision and recall
- **Confusion Matrix**: a full breakdown of correct/incorrect predictions per class

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Environment:** Jupyter Notebook

## 📁 Project Structure

```
├── NOTEBOOK_FILE_UJJWAL_DAHAL.ipynb   # Full implementation: preprocessing, EDA, modeling, evaluation
├── PROJECT_REPORT.pdf                 # Written report with methodology and results
├── processed.cleveland.csv            # Cleaned dataset used for modeling
└── README.md
```

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Heart-Disease-Prediction-Using-3-ML-Algorithms.git
   cd Heart-Disease-Prediction-Using-3-ML-Algorithms
   ```

2. **Install the required libraries**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook NOTEBOOK_FILE_UJJWAL_DAHAL.ipynb
   ```

4. **Run all cells** from top to bottom. The notebook will load the data, preprocess it, train all three models, tune their hyperparameters, and print out the final comparison automatically.

## 📈 Results and Observations

### 10-Fold Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| KNN | 82.84% | 84.25% | 76.98% | 80.45% |
| **Naive Bayes** | **83.83%** | **84.62%** | **79.14%** | **81.78%** |
| Decision Tree | 79.87% | 83.05% | 70.50% | 76.26% |

### Mean Cross-Validation Accuracy (± Standard Deviation)

| Model | Mean Accuracy | Std. Dev |
|---|---:|---:|
| KNN | 82.80% | ±6.16% |
| **Naive Bayes** | **83.78%** | ±6.10% |
| Decision Tree | 79.80% | ±7.92% |

### Key Observations

- **Naive Bayes came out on top** across every metric: accuracy, precision, recall, and F1-score. It also had the lowest standard deviation, meaning its performance was the most *consistent* across the 10 folds, not just the highest on average.
- **KNN was a close second**, performing almost as well as Naive Bayes once properly scaled and tuned (K=13 with Euclidean distance).
- **Decision Tree performed the weakest** of the three, particularly on **recall** (70.50%), meaning it missed more actual heart disease cases than the other two models. Its higher standard deviation (±7.92%) also suggests it was less stable across different data splits, likely because a shallow tree (max_depth=3) is more sensitive to which patients land in which fold.
- Interestingly, **all three models kept all 13 features**, since correlation analysis showed even "weak" features like `fbs` and `chol` still added some value when combined with stronger predictors like `thal` and `ca`.
- Overall, the results suggest that for this dataset and feature set, a simple probabilistic model (Naive Bayes) generalizes slightly better than a distance-based or rule-based model, though the margin between KNN and Naive Bayes is small enough that either would be a reasonable choice in practice.

## 👤 Author

**Ujjwal Dahal**
Department of Electronics and Computer Engineering, IOE Thapathali Campus
📧 ujjwaldahal20@gmail.com

## 🙏 Acknowledgment

This project reimplements the general workflow presented in Ansari et al. (2023). Results differ from the original paper because of differences in preprocessing, feature handling, and experimental setup. The goal here was to reproduce the *methodology*, not match the exact numbers reported in the paper.

One key addition made in this project that wasn't part of the original paper's methodology is **hyperparameter tuning using GridSearchCV**. The original study did not tune model hyperparameters; each model here was tuned before final evaluation to get a fairer, better-optimized comparison between KNN, Naive Bayes, and Decision Tree.
