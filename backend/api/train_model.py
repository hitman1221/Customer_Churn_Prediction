# backend/api/train_model.py

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    RocCurveDisplay
)

# ─── 0) Database Connection ───────────────────────────────────────────────────
DB_USER     = os.getenv('DB_USER', 'user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_HOST     = os.getenv('DB_HOST', 'postgres')
DB_PORT     = os.getenv('DB_PORT', '5432')
DB_NAME     = os.getenv('DB_NAME', 'churn_db')

ENGINE = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ─── 1) Load & Feature–Engineer ───────────────────────────────────────────────
def load_and_engineer() -> pd.DataFrame:
    csv_path = '../data/Telco-Customer-Churn.csv'
    if not os.path.exists(csv_path):
        csv_path = 'data/Telco-Customer-Churn.csv'
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'customerID': 'customer_id',
        'SeniorCitizen': 'senior_citizen',
        'PhoneService': 'phone_service',
        'MultipleLines': 'multiple_lines',
        'InternetService': 'internet_service',
        'OnlineSecurity': 'online_security',
        'OnlineBackup': 'online_backup',
        'DeviceProtection': 'device_protection',
        'TechSupport': 'tech_support',
        'StreamingTV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies',
        'PaperlessBilling': 'paperless_billing',
        'MonthlyCharges': 'monthly_charges',
        'TotalCharges': 'total_charges',
        'PaymentMethod': 'payment_method',
        'Churn': 'churn',
        'Partner': 'partner',
        'Dependents': 'dependents',
        'Contract': 'contract'
    })
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Binary target
    df['churn'] = df['churn'].str.lower().map({'yes':1, 'no':0})
    # Ensure numeric
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

    # Keep base features
    base = [
        'tenure','monthly_charges','total_charges',
        'contract','gender','senior_citizen','partner','dependents',
        'internet_service','multiple_lines','online_security','online_backup',
        'device_protection','tech_support','streaming_tv','streaming_movies',
        'paperless_billing','payment_method'
    ]
    df = df[base + ['churn']].dropna()

    # Engineered features
    df['avg_charge_per_month'] = df['total_charges'] / (df['tenure'] + 1e-3)
    svc_cols = [
        'multiple_lines','online_security','online_backup',
        'device_protection','tech_support',
        'streaming_tv','streaming_movies'
    ]
    df['services_count'] = df[svc_cols].apply(
        lambda row: sum(str(x).lower()=='yes' for x in row),
        axis=1
    )
    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=[-1,12,24,48,np.inf],
        labels=['0-12','12-24','24-48','48+']
    )
    return df

# ─── 2) Split & Preprocess ────────────────────────────────────────────────────
def build_preprocessor():
    # Numeric pipeline
    num_feats = [
        'tenure','monthly_charges','total_charges',
        'avg_charge_per_month','services_count'
    ]
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_feats = [
        'contract','gender','senior_citizen','partner','dependents',
        'internet_service','paperless_billing','payment_method',
        'tenure_bucket'
    ]
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_pipe, num_feats),
        ('cat', cat_pipe, cat_feats)
    ])

# ─── 3) Train + Hyperparameter Tuning ─────────────────────────────────────────
def train_and_tune(X_train, y_train):
    preprocessor = build_preprocessor()

    # Full pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    # Hyperparameter space (20 random draws)
    param_dist = {
        'model__n_estimators':     [100, 200, 300, 500],
        'model__learning_rate':    [0.01, 0.03, 0.05, 0.1],
        'model__max_depth':        [3, 5, 7],
        'model__subsample':        [0.6, 0.8, 1.0],
        'model__min_samples_split':[2, 5, 10],
        'model__min_samples_leaf': [1, 3, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    search.fit(X_train, y_train)
    print("▶️ Best GBM params:", search.best_params_)
    print("▶️ Best CV ROC AUC:", search.best_score_)
    return search.best_estimator_

# ─── 4) Threshold Optimization ────────────────────────────────────────────────
def find_best_threshold(y_true, y_proba):
    best_thr, best_f1 = 0.5, 0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

# ─── 5) Evaluate & Report ────────────────────────────────────────────────────
def evaluate_and_report(model, X_test, y_test, threshold):
    os.makedirs('reports', exist_ok=True)

    proba = model.predict_proba(X_test)[:,1]
    preds = (proba >= threshold).astype(int)

    # Classification report
    cr = classification_report(y_test, preds)
    with open('reports/classification_report.txt','w', encoding='utf-8') as f:
        f.write(cr)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Stay','Churn'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['Stay','Churn'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    for i in (0,1):
        for j in (0,1):
            ax.text(j, i, cm[i,j], ha='center', va='center')
    fig.savefig('reports/confusion_matrix.png')
    plt.close(fig)

    # ROC curve
    fig2, ax2 = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax2)
    fig2.savefig('reports/roc_curve.png')
    plt.close(fig2)

    # Markdown summary
    auc = roc_auc_score(y_test, proba)
    f1v = f1_score(y_test, preds)
    md = f"""\
# Gradient Boosting Churn Model Report

**Test ROC AUC:** {auc:.4f}  
**Test F1 (@thr={threshold:.2f}):** {f1v:.3f}

## Classification Report  
{cr}

scss
Copy
Edit

![](confusion_matrix.png) ![](roc_curve.png)
"""
    with open('reports/model_report.md','w', encoding='utf-8') as f:
        f.write(md)

# ─── 6) Main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load & engineer
    df = load_and_engineer()

    # Split
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train & tune
    print("🔍 Tuning Gradient Boosting...")
    gbm_pipe = train_and_tune(X_train, y_train)

    # Threshold tuning
    print("🔧 Finding best classification threshold...")
    proba_test = gbm_pipe.predict_proba(X_test)[:,1]
    thr, f1v = find_best_threshold(y_test, proba_test)
    print(f"👉 Best threshold = {thr:.2f}, F1 = {f1v:.3f}")

    # Evaluate & save
    print("📊 Generating reports...")
    evaluate_and_report(gbm_pipe, X_test, y_test, thr)

    # Persist model
    with open('model.pkl','wb') as f:
        pickle.dump(gbm_pipe, f)
    print("✅ Done. Model and reports are saved.")