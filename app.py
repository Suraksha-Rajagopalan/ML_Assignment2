
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Student Academic Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

TARGET_COLUMN = "Target"

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #ecd8c1;
        text-align: center;
    }

    .sub-title {
        text-align: center;
        color: #04748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        color: ##25b6eb;
        font-size: 23px;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():

    loaded_models = {}

    possible_directories = [
        BASE_DIR / "models",
        BASE_DIR / "model"
    ]

    model_directory = next(
        (folder for folder in possible_directories if folder.exists()),
        None
    )

    if model_directory is None:
        return loaded_models, ["Model directory not found."]

    errors = []

    for model_name, filename in MODEL_FILES.items():

        model_path = model_directory / filename

        if model_path.exists():

            try:
                loaded_models[model_name] = joblib.load(model_path)

            except Exception as error:
                errors.append(
                    f"{model_name}: {str(error)}"
                )

        else:
            errors.append(f"Missing model file: {filename}")

    return loaded_models, errors


models, model_errors = load_models()

def calculate_auc(model, X_test, y_test):

    classes = model.classes_

    if hasattr(model, "predict_proba"):

        scores = model.predict_proba(X_test)

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(X_test)

    else:
        return np.nan

    return roc_auc_score(
        y_test,
        scores,
        labels=classes,
        multi_class="ovr",
        average="macro"
    )


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    try:
        auc = calculate_auc(model, X_test, y_test)

    except (ValueError, AttributeError):
        auc = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc,
        "Precision": precision_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions)
    }

    return predictions, metrics


st.markdown(
    '<div class="main-title">🎓 Student Academic Success Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Machine Learning Based Prediction of Student Dropout '
    'and Academic Success'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.markdown("""
### About This Project

This application demonstrates five machine learning classification
algorithms trained on the **Predict Students' Dropout and Academic
Success** dataset. The application evaluates previously trained models using an
uploaded test dataset.
**Target classes:**
- Graduate
- Dropout
- Enrolled

No model is retrained when a test dataset is uploaded.
""")

st.sidebar.title("🎛️ Model Configuration")

st.sidebar.markdown(
    "Select a trained classifier to inspect its performance."
)

if not models:

    st.error(
        "No trained models could be loaded. "
        "Please check your models folder."
    )

    st.stop()

if model_errors:

    with st.expander("Model loading warnings"):
        for error in model_errors:
            st.warning(error)


selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

st.sidebar.divider()

st.sidebar.markdown("**Available Models**")

for model_name in models:
    st.sidebar.write(f"✓ {model_name}")


st.markdown(
    '<div class="section-title">📂 Upload Test Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your test data(in CSV format) file",
    type=["csv"],
    help="CSV must contain all 36 input features and the Target column."
)

if uploaded_file is None:

    st.info(
        "Please upload your test data(in CSV format) file to evaluate "
        "the trained classification models."
    )
    st.stop()


try:

    test_data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read CSV file: {error}")
    st.stop()


if TARGET_COLUMN not in test_data.columns:

    st.error(
        "The uploaded CSV must contain the 'Target' column "
        "for evaluation."
    )

    st.stop()


# Identify the expected features from the saved pipeline.

first_model = next(iter(models.values()))

if hasattr(first_model, "feature_names_in_"):

    expected_features = list(first_model.feature_names_in_)

else:

    expected_features = [
        column for column in test_data.columns
        if column != TARGET_COLUMN
    ]


missing_features = [
    feature for feature in expected_features
    if feature not in test_data.columns
]

if missing_features:

    st.error(
        "Missing required input features: "
        + ", ".join(missing_features)
    )

    st.stop()


if test_data.empty:

    st.error("The uploaded dataset is empty.")
    st.stop()


X_test = test_data[expected_features]
y_test = test_data[TARGET_COLUMN]


if y_test.isna().any():

    st.error("The Target column contains missing values.")
    st.stop()


st.success("Test dataset uploaded and validated successfully!")

with st.expander("Preview Uploaded Dataset"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )

@st.cache_data
def run_all_evaluations(_models, X, y):

    all_results = {}
    all_predictions = {}

    for name, model in _models.items():

        try:

            predictions, metrics = evaluate_model(
                model, X, y
            )

            all_results[name] = metrics
            all_predictions[name] = predictions

        except Exception as error:

            all_results[name] = {
                metric: np.nan
                for metric in [
                    "Accuracy", "AUC", "Precision",
                    "Recall", "F1 Score", "MCC"
                ]
            }

            all_predictions[name] = None

            st.warning(
                f"Could not evaluate {name}: {error}"
            )

    return all_results, all_predictions


results, predictions_dict = run_all_evaluations(
    models,
    X_test,
    y_test
)

st.divider()

st.markdown(
    '<div class="section-title">📈 Model Evaluation Results</div>',
    unsafe_allow_html=True
)

selected_metrics = results[selected_model_name]
selected_predictions = predictions_dict[selected_model_name]

if selected_predictions is None:

    st.error("Selected model could not be evaluated.")
    st.stop()


st.subheader(f"Performance: {selected_model_name}")

metric_columns = st.columns(6)

metric_names = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

for column, metric in zip(metric_columns, metric_names):

    value = selected_metrics[metric]

    column.metric(
        metric,
        f"{value:.4f}" if pd.notna(value) else "N/A"
    )

st.divider()

st.markdown(
    '<div class="section-title">🔍 Confusion Matrix</div>',
    unsafe_allow_html=True
)

classes = models[selected_model_name].classes_

cm = confusion_matrix(
    y_test,
    selected_predictions,
    labels=classes
)

fig, ax = plt.subplots(figsize=(3.5, 2.8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    linewidths=0.5,
    ax=ax
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(f"Confusion Matrix - {selected_model_name}")

st.pyplot(fig)
plt.close(fig)

st.divider()

st.markdown(
    '<div class="section-title">📋 Classification Report</div>',
    unsafe_allow_html=True
)

report = classification_report(
    y_test,
    selected_predictions,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)
st.divider()

st.markdown(
    '<div class="section-title">🏆 Model Comparison</div>',
    unsafe_allow_html=True
)

comparison_df = pd.DataFrame(results).transpose()

comparison_df.index.name = "ML Model"

st.dataframe(
    comparison_df.style.format("{:.4f}"),
    use_container_width=True
)


st.subheader("Performance Comparison")

available_metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

selected_comparison_metric = st.selectbox(
    "Choose metric for comparison",
    available_metrics
)

chart_data = comparison_df[
    [selected_comparison_metric]
].dropna()

st.bar_chart(
    chart_data,
    use_container_width=True
)


st.divider()

st.markdown(
    '<div class="section-title">🥇 Overall Best Model</div>',
    unsafe_allow_html=True
)

valid_auc = comparison_df["AUC"].dropna()

if not valid_auc.empty:

    best_model_name = valid_auc.idxmax()
    best_auc = valid_auc.max()

    col1, col2 = st.columns(2)

    col1.metric("Best Model", best_model_name)
    col2.metric("AUC Score", f"{best_auc:.4f}")

    st.success(
        f"{best_model_name} achieved the highest "
        f"AUC score of {best_auc:.4f} on the uploaded test data."
    )


st.divider()

st.markdown(
    '<div class="section-title">📌 Prediction Distribution</div>',
    unsafe_allow_html=True
)

distribution = pd.Series(
    selected_predictions
).value_counts()

fig, ax = plt.subplots(figsize=(3.5, 2.8))

distribution.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    f"Predicted Student Outcomes - {selected_model_name}"
)

ax.set_xlabel("Predicted Outcome")
ax.set_ylabel("Number of Students")

plt.xticks(rotation=0)

st.pyplot(fig)
plt.close(fig)

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray; padding:15px;">
        <b>Student Academic Success Prediction System</b><br>
        M.Tech (AIML) | Machine Learning Assignment 2<br>
        Developed using Python, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
