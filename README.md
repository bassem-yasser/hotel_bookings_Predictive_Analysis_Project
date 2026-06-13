<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2C3E50,100:3498DB&height=220&section=header&text=Hotel%20Booking%20Cancellation%20Predictor&fontSize=34&fontColor=ffffff&fontAlignY=40&desc=End-to-End%20Predictive%20Analysis%20Pipeline%20%7C%20119%2C390%20Records&descAlignY=60&descSize=15" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SMOTE](https://img.shields.io/badge/SMOTE-Balanced-2ECC71?style=for-the-badge)](https://imbalanced-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretability-9B59B6?style=for-the-badge)](https://shap.readthedocs.io)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
[![License](https://img.shields.io/badge/License-MIT-27AE60?style=for-the-badge)](LICENSE)

<br/>

> **🏨 Predicting hotel booking cancellations using a full ML pipeline — EDA → Preprocessing → Feature Engineering → Model Training → Interpretation — on 119,390 real-world records.**

<br/>

[📊 Project Phases](#-project-phases) · [📈 Model Results](#-model-results) · [🚀 Quick Start](#-quick-start) · [📁 Structure](#-repository-structure) · [👥 Team](#-team)

</div>

---

## 📌 Table of Contents

- [🏨 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🔬 Project Phases](#-project-phases)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Quick Start](#-quick-start)
- [📈 Model Results](#-model-results)
- [📁 Repository Structure](#-repository-structure)
- [👥 Team](#-team)

---

## 🏨 Project Overview

Hotel booking cancellations cause significant revenue loss and operational disruption across the hospitality industry. This project builds a complete **binary classification pipeline** to predict whether a hotel booking will be cancelled before arrival — enabling hotels to act proactively on at-risk reservations.

```
🎯 Target Variable   →  is_canceled         (0 = Kept,  1 = Cancelled)
📦 Dataset Size      →  119,390 rows  ×  32 columns
🔁 Task Type         →  Binary Classification
🏆 Best Model        →  XGBoost (Tuned)  —  Recall: 82%  |  F1: 0.71
📐 Train / Test      →  80% / 20%  (Stratified Split)
⚖️ Class Balancing   →  SMOTE  (50/50 after resampling)
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |
| **Total Records** | 119,390 bookings |
| **Features (original)** | 32 columns (numeric + categorical) |
| **Features (engineered)** | 53 columns after feature engineering |
| **Features (after Lasso)** | 50 columns retained |
| **Target** | `is_canceled` (0 = Not Cancelled, 1 = Cancelled) |
| **Hotels** | Resort Hotel & City Hotel |
| **Class Balance (raw)** | 72.48% Not Cancelled · 27.52% Cancelled |
| **Class Balance (post-SMOTE)** | 50% · 50% |

### Key Feature Groups

| Category | Features |
|---|---|
| 📅 **Booking Info** | `lead_time`, `arrival_date_*`, `booking_changes`, `days_in_waiting_list` |
| 🏠 **Stay Details** | `stays_in_weekend_nights`, `stays_in_week_nights`, `reserved_room_type`, `assigned_room_type` |
| 👥 **Guest Profile** | `adults`, `children`, `babies`, `customer_type`, `is_repeated_guest` |
| 💳 **Commercial** | `deposit_type`, `market_segment`, `distribution_channel`, `adr` |
| 📋 **History** | `previous_cancellations`, `previous_bookings_not_canceled` |
| 🛠️ **Engineered** | `total_guests`, `total_stay_nights`, `total_revenue`, `is_room_changed`, `has_children`, `is_weekend_stay`, `has_special_requests`, `lead_time_category`, `price_tier`, `season`, `is_peak_season` |

---

## 🔬 Project Phases

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ARCHITECTURE                             │
│                                                                          │
│   Phase 1          Phase 2            Phase 3          Phase 4          │
│  ┌────────┐       ┌─────────────┐    ┌──────────┐    ┌─────────────┐   │
│  │  EDA   │──────►│Preprocessing│───►│ Feature  │───►│   Model     │   │
│  │(Before)│       │& Engineering│    │Selection │    │  Training   │   │
│  └────────┘       └─────────────┘    └──────────┘    └──────┬──────┘   │
│                                                              │           │
│   Phase 5                  Phase 3 (Post-Clean)             ▼           │
│  ┌─────────────────────┐   ┌────────────────┐    ┌──────────────────┐  │
│  │ Model Interpretation│   │  EDA (After    │◄───│  Evaluation &    │  │
│  │   (SHAP Values)     │   │   Cleaning)    │    │  Comparison      │  │
│  └─────────────────────┘   └────────────────┘    └──────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### 🔍 Phase 1 — Exploratory Data Analysis (Before Cleaning)

> *"Get to know your data before touching any models."*

- Loaded **Hotel Booking Demand** dataset: **119,390 rows × 32 columns**
- Identified `company` column as highest missing-value column
- Performed **Univariate Analysis** — distributions for all numeric & categorical features
- **Bivariate Analysis** — cancellation rates broken down by hotel type, market segment, deposit type
- **Correlation Analysis** — heatmap revealed weak linear correlations among numeric features
- **Outliers Detection** — extreme values found in `lead_time` and `adr`
- Key finding: **City Hotels show higher cancellation rates than Resort Hotels**

---

### ⚙️ Phase 2 — Preprocessing & Feature Engineering

> *"Prepare your data for modeling."*

**Step 1 — Data Cleaning:**
- Detected and removed **duplicate rows**
- Removed invalid/noise records (e.g. bookings with 0 adults, 0 children, 0 babies)
- Handled missing values via **KNN Imputer** and **Simple Imputer**
- Corrected data types for date and categorical columns
- Applied **Winsorization** to cap outliers in skewed features

**Step 2 — Feature Engineering (53 final columns):**

| New Feature | Description |
|---|---|
| `total_guests` | adults + children + babies |
| `total_stay_nights` | weekend nights + week nights |
| `total_revenue` | adr × total stay nights |
| `is_room_changed` | reserved room ≠ assigned room |
| `has_children` | binary flag for children/babies |
| `is_weekend_stay` | flag for weekend-only stays |
| `has_special_requests` | binary from total_of_special_requests |
| `lead_time_category` | binned: Same-Day / Short / Medium / Long / Very Long |
| `stay_duration_category` | binned stay length |
| `price_tier` | quartile-based ADR category |
| `customer_segment` | repeat vs new guest |
| `booking_type` | derived from market + deposit |
| `season` | Spring / Summer / Autumn / Winter |
| `is_peak_season` | July–August flag |
| `month_sin / month_cos` | cyclical encoding of arrival month |

**Step 3 — Data Splitting:**
- **80 / 20 stratified split** → X_train: 69,776 · X_test: 17,446
- Target distribution preserved across splits

**Step 4 — Data Transformation:**
- Applied **StandardScaler** / **RobustScaler** to numeric features
- **OneHotEncoder** for nominal categories
- **OrdinalEncoder** for ordered categories
- **LabelEncoder** for binary features

**Step 5 — Class Balancing with SMOTE:**

```
BEFORE SMOTE:
  Not Canceled (0): 72.48%
  Canceled    (1): 27.52%

AFTER SMOTE (training only):
  Not Canceled (0): 50.00%    X_train shape: (101,152 × 53)
  Canceled    (1): 50.00%    X_test  shape:  (17,446 × 53)  ← untouched
```

---

### 🔍 Phase 3 — EDA After Cleaning

- Re-visualized all distributions on the cleaned, engineered dataset
- Confirmed relationships: `lead_time`, `deposit_type`, `previous_cancellations`, and `total_revenue` are strongest visual predictors
- Verified no data leakage in engineered features

---

### 🎯 Phase 4 — Feature Selection

**Method: Lasso (L1) Regularization via `SelectFromModel`**

- Applied `LogisticRegression(penalty='l1', C=0.1)` as feature selector
- **Original features:** 53
- **Features dropped:** 3
- **Features retained:** 50

**Top Features Kept by Lasso:**

```
arrival_date_day_of_month    is_repeated_guest         previous_cancellations
previous_bookings_not_canceled   booking_changes       days_in_waiting_list
required_car_parking_spaces  total_guests              total_stay_nights
total_revenue                is_room_changed           has_children
is_weekend_stay              has_special_requests      is_peak_season
month_sin / month_cos        country_encoded           hotel_Resort Hotel
meal_type (all dummies)      market_segment (all)      deposit_type (all)
customer_type (all)          lead_time_category (all)  + more...
```

---

### 🤖 Phase 5 — Model Training & Evaluation

Four models were trained and compared on the same test set (17,446 samples):

| # | Model | Configuration |
|---|---|---|
| 1 | **Logistic Regression (Baseline)** | Default · no regularization |
| 2 | **Logistic Regression + GridSearchCV** | 5-fold CV · L1/L2 · C ∈ {0.01, 0.1, 1.0, 10.0} · 8 candidates · 40 fits |
| 3 | **Logistic Regression + Lasso (L1)** | `SelectFromModel` · C=0.1 · 50 features |
| 4 | **Random Forest** | 100 trees · `n_jobs=-1` · `random_state=42` |
| 5 | **XGBoost** | 200 estimators · `lr=0.1` · `max_depth=6` |
| 6 | **XGBoost (Tuned) ⭐** | 300 estimators · `lr=0.05` · `max_depth=10` · `scale_pos_weight` |

---

### 🧠 Phase 6 — Model Interpretation

- **Random Forest feature importances** plotted: Top 10 Drivers of Hotel Cancellations
- **SHAP (SHapley Additive exPlanations)** applied to the best model
- Waterfall and beeswarm plots generated for individual and global explanations
- Confirmed: `lead_time`, `deposit_type`, `previous_cancellations`, and `total_revenue` are the top SHAP contributors

---

## 📈 Model Results

### Full Performance Comparison — All Models

> All evaluated on the **same held-out test set: 17,446 samples** (never seen during training or SMOTE)

| Model | Accuracy | F1 (Cancelled) | Precision (Cancelled) | Recall (Cancelled) | F1 (Weighted) |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression (Baseline) | 82.69% | 0.66 | 0.72 | 0.60 | 0.82 |
| Logistic Regression + Lasso (L1) | 81.05% | 0.61 | 0.70 | 0.54 | 0.80 |
| Random Forest (100 trees) | 83.22% | 0.67 | 0.73 | 0.62 | 0.83 |
| XGBoost (default) | 83.40% | 0.67 | 0.73 | 0.62 | 0.83 |
| **XGBoost (Tuned) ⭐ BEST** | **81.82%** | **0.71** | **0.63** | **0.82** | **0.84** |

> **Why is XGBoost Tuned the best model?** While its raw accuracy (81.82%) is slightly lower than default XGBoost (83.40%), the tuned model was specifically optimized to **maximize recall on cancellations** — it catches **82% of all actual cancellations** vs only 62% in the default version. In a hotel revenue management context, missing a true cancellation (false negative) is far more costly than a false alarm.

---

### 🏆 Best Model — XGBoost (Tuned) — Full Report

**Configuration:**
```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=weight_ratio,  # Prioritizes minority class (Cancelled)
    random_state=42,
    n_jobs=-1
)
```

**Classification Report:**
```
               precision    recall  f1-score   support

 Not Cancelled    0.92      0.82      0.87     12,644
    Cancelled    0.63      0.82      0.71      4,802

     accuracy                          0.82     17,446
    macro avg       0.78      0.82      0.79     17,446
 weighted avg       0.84      0.82      0.82     17,446
```

---

### 📊 Model-by-Model Detailed Breakdown

<details>
<summary><b>📋 Logistic Regression (Baseline)</b></summary>

```
Train Accuracy : 82.83%
Test Accuracy  : 82.69%

               precision    recall  f1-score   support

           0       0.86      0.91      0.88     12,644
           1       0.72      0.60      0.66      4,802

    accuracy                           0.83     17,446
   macro avg       0.79      0.76      0.77     17,446
weighted avg       0.82      0.83      0.82     17,446
```
</details>

<details>
<summary><b>📋 Logistic Regression + Lasso Feature Selection</b></summary>

```
Features: 53 → 50 (dropped 3 via L1 regularization)
Test Accuracy : 81.05%

               precision    recall  f1-score   support

           0       0.84      0.91      0.87     12,644
           1       0.70      0.54      0.61      4,802

    accuracy                           0.81     17,446
   macro avg       0.77      0.73      0.74     17,446
weighted avg       0.80      0.81      0.80     17,446
```
</details>

<details>
<summary><b>📋 Random Forest (100 Trees)</b></summary>

```
Test Accuracy : 83.22%

               precision    recall  f1-score   support

           0       0.86      0.91      0.89     12,644
           1       0.73      0.62      0.67      4,802

    accuracy                           0.83     17,446
   macro avg       0.80      0.77      0.78     17,446
weighted avg       0.83      0.83      0.83     17,446
```
</details>

<details>
<summary><b>📋 XGBoost (Default)</b></summary>

```
Accuracy : 83.40%   |   F1-Score: 0.6735

               precision    recall  f1-score   support

           0       0.86      0.91      0.89     12,644
           1       0.73      0.62      0.67      4,802

    accuracy                           0.83     17,446
   macro avg       0.80      0.77      0.78     17,446
weighted avg       0.83      0.83      0.83     17,446
```
</details>

<details>
<summary><b>📋 XGBoost (Tuned) — BEST MODEL ⭐</b></summary>

```
Accuracy : 81.82%   |   F1-Score: 0.7121

               precision    recall  f1-score   support

           0       0.92      0.82      0.87     12,644
           1       0.63      0.82      0.71      4,802

    accuracy                           0.82     17,446
   macro avg       0.78      0.82      0.79     17,446
weighted avg       0.84      0.82      0.82     17,446
```
</details>

---

### 🔑 Key EDA Insights

```
1. Missing values were successfully handled (KNN + Simple Imputer)
2. Duplicate rows were detected and removed
3. Lead time has a strong relationship with cancellation probability
4. City Hotels show significantly higher cancellation rates than Resort Hotels
5. Some numerical features retained slight skewness after winsorization
6. Most numerical variables show weak linear correlations with each other
7. Dataset cleaned, engineered (53 features), and ready for ML modeling
```

---

## ⚙️ Tech Stack

| Category | Libraries / Tools |
|---|---|
| **Language** | Python 3.12 |
| **Data Manipulation** | `pandas` · `numpy` |
| **Visualization** | `matplotlib` · `seaborn` |
| **Preprocessing** | `scikit-learn` (StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, KNNImputer) |
| **Class Balancing** | `imbalanced-learn` (SMOTE) |
| **Feature Selection** | `scikit-learn` (SelectFromModel, Lasso L1) |
| **Modeling** | `scikit-learn` · `xgboost` · `lightgbm` |
| **Tuning** | `GridSearchCV` · `RandomizedSearchCV` · `StratifiedKFold` |
| **Interpretation** | `shap` |
| **Pipeline** | `sklearn.pipeline.Pipeline` |
| **Data Source** | `kagglehub` |
| **Environment** | Google Colab / Jupyter Notebook |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/hotel-booking-cancellation.git
cd hotel-booking-cancellation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn \
            xgboost lightgbm shap kagglehub jupyter
```

### 3. Set Up Kaggle API *(for automatic dataset download)*
```bash
# 1. Go to https://www.kaggle.com/settings → API → Create New Token
# 2. Download kaggle.json and place it at:
#    Linux/Mac → ~/.kaggle/kaggle.json
#    Windows   → C:\Users\<YourName>\.kaggle\kaggle.json
chmod 600 ~/.kaggle/kaggle.json   # Linux/Mac only
```

### 4. Launch the Notebook
```bash
jupyter notebook PA_Project.ipynb
```

> **Alternative:** Place `hotel_bookings.csv` directly in the project folder and update the data-loading cell to read from local disk instead of kagglehub.

### 5. Run All Cells
The notebook runs **end-to-end** without errors. Phases are clearly labeled with markdown headings.

---

## 📁 Repository Structure

```
hotel-booking-cancellation/
│
├── 📓 PA_Project.ipynb                    # Main notebook — full pipeline (Phases 1–10)
├── 📄 report.pdf                          # Written project report (≤12 pages)
├── 📊 hotel_bookings.csv                  # Raw dataset (119,390 records × 32 features)
├── 📋 README.md                           # This file
└── 📦 requirements.txt                    # Python dependencies
```

---

## 👥 Team

| Member | Student ID | Responsibilities |
|---|---|---|
| Marim| — | EDA (Before & After Cleaning), Visualizations |
| Abdelrahman Ali| — | Preprocessing, Feature Engineering, SMOTE |
| Mohamed Saber | — | Feature Selection (Lasso), Model Training |
| Bassem Yasser and Ahmed Mahmoued | — | Model Tuning, Evaluation, SHAP Interpretation, Report |

---

<div align="center">

### ⭐ Star this repo if you found it useful!

*Built as a final project for a Predictive Analysis course.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3498DB,100:2C3E50&height=120&section=footer" width="100%"/>

</div>
