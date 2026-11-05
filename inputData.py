
# ============================================================
# STUDENT DROPOUT AND ACADEMIC SUCCESS CLASSIFICATION
# UCI Dataset ID: 697
# ============================================================

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from scipy.stats import randint, loguniform

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report
)

warnings.filterwarnings("ignore")

# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "data.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# =====================================================
# 2. LOAD DATASET
# =====================================================

print("Loading Student Dropout dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Download and extract the UCI dataset into the data folder."
    )

# Important: UCI dataset uses semicolon delimiter
df = pd.read_csv(DATASET_PATH, sep=";")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# =====================================================
# 3. DATA INSPECTION
# =====================================================

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nMissing values:", df.isnull().sum().sum())

print("\nDuplicate records:", df.duplicated().sum())

# =====================================================
# 4. SEPARATE FEATURES AND TARGET
# =====================================================

TARGET_COLUMN = "Target"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found!"
    )

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print("\nNumber of features:", X.shape[1])
print("Number of target classes:", y.nunique())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget distribution (%):")
print((y.value_counts(normalize=True) * 100).round(2))

# =====================================================
# 5. TRAIN-TEST SPLIT
# =====================================================

print("\nPerforming stratified 80:20 split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# =====================================================
# 6. SAVE ORIGINAL, TRAIN AND TEST DATA
# =====================================================

# Save original dataset
df.to_csv(
    os.path.join(DATA_DIR, "original_dataset.csv"),
    index=False
)

# Save training dataset
train_df = X_train.copy()
train_df[TARGET_COLUMN] = y_train

train_df.to_csv(
    os.path.join(DATA_DIR, "train_data.csv"),
    index=False
)

# Save testing dataset
test_df = X_test.copy()
test_df[TARGET_COLUMN] = y_test

test_df.to_csv(
    os.path.join(DATA_DIR, "test_data.csv"),
    index=False
)

# =====================================================
# 7. DISPLAY SPLIT INFORMATION
# =====================================================

print("\n" + "=" * 55)
print("DATASET SPLIT SUMMARY")
print("=" * 55)

print("Original records:", len(df))
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nSaved files:")
print("1. data/original_dataset.csv")
print("2. data/train_data.csv")
print("3. data/test_data.csv")

print("\nDataset preparation completed successfully!")

# ============================================================
# 5. FEATURE PREPROCESSING
# ============================================================

# These columns contain category codes rather than
# continuous measurements. Treat them as categorical.

categorical_columns = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]

# Keep only columns that actually exist
categorical_columns = [
    col for col in categorical_columns
    if col in X_train.columns
]

numerical_columns = [
    col for col in X_train.columns
    if col not in categorical_columns
]

print("\nCategorical features:", len(categorical_columns))
print("Numerical features:", len(numerical_columns))


# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)

# Model training configuration
CV_FOLDS = 5
N_ITER = 8

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# 6. DEFINE FIVE CLASSIFICATION MODELS
# ============================================================

models = {

    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "params": {
            "classifier__C": loguniform(0.01, 100),
            "classifier__solver": ["lbfgs"]
        }
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "params": {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 5, 10, 15, 20],
            "classifier__min_samples_split": randint(2, 20),
            "classifier__min_samples_leaf": randint(1, 10)
        }
    },

    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "classifier__n_neighbors": randint(3, 30),
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        }
    },

    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {
            "classifier__var_smoothing": loguniform(
                1e-12, 1e-7
            )
        }
    },

    "Support Vector Machine": {
        "model": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "params": {
            "classifier__C": loguniform(0.1, 100),
            "classifier__gamma": ["scale", "auto"]
        }
    }
}


# ============================================================
# 7. CROSS-VALIDATION CONFIGURATION
# ============================================================

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 8. TRAIN, OPTIMIZE AND EVALUATE
# ============================================================

results = []
best_auc = -1
best_model_name = None

best_parameters = {}

for model_name, configuration in models.items():

    print("\n" + "=" * 65)
    print("Training:", model_name)
    print("=" * 65)

    # Build pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", configuration["model"])
    ])

    # Hyperparameter optimization
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=configuration["params"],
        n_iter=N_ITER,
        scoring="roc_auc_ovr",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    # Fit using training data only
    search.fit(X_train, y_train)

    # Get optimized model
    optimized_model = search.best_estimator_

    # Predict test classes
    y_pred = optimized_model.predict(X_test)

    # Predict class probabilities
    y_proba = optimized_model.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_proba,
        multi_class="ovr",
        average="macro",
        labels=optimized_model.classes_
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Store evaluation results
    result = {
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "Best CV AUC": search.best_score_
    }

    results.append(result)

    # Save optimized model
    model_filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(
        optimized_model,
        os.path.join(MODEL_DIR, model_filename)
    )

    best_parameters[model_name] = search.best_params_

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nEvaluation metrics:")
    for metric, value in result.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    # Identify overall winner using test cross-validation AUC
    if search.best_score_ > best_auc:
        best_auc = search.best_score_
        best_model_name = model_name
        best_model = optimized_model


# ============================================================
# 9. SAVE EVALUATION RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="AUC",
    ascending=False
)

results_df.to_csv(
    os.path.join(RESULT_DIR, "evaluation_metrics.csv"),
    index=False
)

# Save hyperparameters
with open(
    os.path.join(RESULT_DIR, "best_parameters.json"),
    "w"
) as file:
    json.dump(
        best_parameters,
        file,
        indent=4,
        default=str
    )


# ============================================================
# 10. SAVE OVERALL BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "best_model.joblib")
)

with open(
    os.path.join(RESULT_DIR, "best_model_info.json"),
    "w"
) as file:

    json.dump({
    "best_model": best_model_name,
    "best_cv_auc": float(best_auc),
    "target_column": "Target",
    "classes": [str(c) for c in best_model.classes_]
}, file, indent=4)


# ============================================================
# 11. DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(results_df.to_string(index=False))

print("\n" + "=" * 75)
print("OVERALL BEST MODEL:", best_model_name)
print("BEST CROSS-VALIDATION AUC:", round(best_auc, 4))
print("=" * 75)

print("\nAll models and results saved successfully!")
