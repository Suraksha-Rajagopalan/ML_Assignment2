# Student Academic Success Predictor

## A. Problem Statement

The objective of this project is to develop a machine learning-based classification system to predict students' academic outcomes using their academic, demographic, and related personal information.

The system classifies students into one of three categories:

- **Graduate:** Students who successfully complete their academic programme.
- **Dropout:** Students who discontinue their academic programme.
- **Enrolled:** Students who are currently continuing their studies.

The project aims to evaluate and compare different machine learning classification algorithms to identify the model that performs best on the selected dataset.

A Streamlit-based web application is also developed to provide an interactive interface where users can upload test datasets, select trained machine learning models, generate predictions, and visualize model performance.

## B. Dataset Description

**Dataset Name:** Predict Students' Dropout and Academic Success

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

**Dataset ID:** 697

The selected dataset contains information about students, including demographic characteristics, academic performance, and socioeconomic factors. It is designed to support the prediction of students' academic outcomes.

**Target Variable:** Target

The target variable contains three classes:

1. Graduate
2. Dropout
3. Enrolled

**Problem Type:** Multi-class Classification

The dataset is used to train and evaluate five different machine learning classification algorithms. Data preprocessing and appropriate feature transformations are applied through machine learning pipelines before model training.

## C. GitHub Repository Link

**GitHub Repository:** https://github.com/Suraksha-Rajagopalan/ML_Assignment2
**StreamLit App**: https://suraksha-rajagopalan-ml-assignment2-app-uxwwc6.streamlit.app/

The repository contains the trained machine learning models, Streamlit application, sample test dataset, and files required to install and execute the project.

## D. Models Used

Five machine learning classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation Metrics

The performance of each model is evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified instances out of the total instances.
- **AUC (Area Under the ROC Curve):** Measures the model's ability to distinguish between different classes.
- **Precision:** Measures the proportion of correctly predicted positive instances among all instances predicted as positive.
- **Recall:** Measures the proportion of actual positive instances correctly identified by the model.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **MCC (Matthews Correlation Coefficient):** Measures the quality of classification predictions by considering true and false positives and negatives.

### Model Performance Comparison

The following table presents the evaluation results obtained for all five classification models.

<img width="1086" height="381" alt="image" src="https://github.com/user-attachments/assets/cfe83409-b96f-4449-a35d-6fc05c62c0d7" />


### Observations on Model Performance

| ML Model Name | Observation About Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved an accuracy of **72.54%** and an AUC of **0.8812**, demonstrating strong classification performance. It obtained a precision of 0.6975, recall of 0.7025, and F1 score of 0.6893. Its MCC of 0.5759 indicates a good correlation between actual and predicted classifications. It performed consistently and was the second-best model overall, showing that a linear classification approach can effectively capture patterns in the dataset. |
| **Decision Tree** | Decision Tree achieved an accuracy of **65.76%** and an AUC of **0.8415**. Its precision was 0.6669, recall was 0.6570, and F1 score was 0.6369. The MCC of 0.5009 indicates moderate classification performance. Although it can capture non-linear relationships between student attributes, its overall performance was lower than SVM and Logistic Regression. |
| **KNN** | K-Nearest Neighbors achieved an accuracy of **71.75%** and an AUC of **0.8424**. It obtained a precision of 0.6902, recall of 0.5940, and F1 score of 0.6038. Its MCC was 0.5352. Although its accuracy was relatively high, its comparatively lower recall indicates that it was less effective at identifying all instances belonging to the actual classes. |
| **Naive Bayes** | Naive Bayes recorded the lowest performance among all five models, with an accuracy of **25.08%** and an AUC of **0.6860**. Its precision was 0.4806, recall was 0.3983, and F1 score was 0.2266. The MCC of 0.1315 also indicates weak agreement between actual and predicted classifications. The conditional independence assumption of Naive Bayes may limit its ability to model relationships among the dataset's features. |
| **Support Vector Machine** | Support Vector Machine achieved the highest overall performance, with an accuracy of **74.35%**, AUC of **0.8862**, precision of **0.7115**, recall of **0.7140**, and F1 score of **0.7041**. It also achieved the highest MCC of **0.5987**. Furthermore, it obtained the best cross-validation AUC of **0.8936**, indicating strong and consistent classification performance across validation folds. It outperformed the other four models across all the reported evaluation metrics. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM)** is the overall winner for the selected dataset.

SVM achieved the highest accuracy, AUC, precision, recall, F1 score, and Matthews Correlation Coefficient among the five evaluated models. It also recorded the highest cross-validation AUC of **0.8936**.

Logistic Regression was the second-best model, with an accuracy of 72.54% and an AUC of 0.8812, making it a competitive alternative.

Therefore, Support Vector Machine was selected as the best-performing model and saved as `best_model.joblib` for use in the Streamlit application.

### Overall Winner for the Dataset

**Best Performing Model: SVM**

Based on the evaluation metrics, **SVM** achieved the strongest overall performance on the selected dataset.

The model was selected by considering its performance across accuracy, AUC, precision, recall, F1 score, and MCC. These metrics provide a broader understanding of classification performance rather than relying on accuracy alone.

The selected model is also saved as `best_model.joblib` and can be used through the Streamlit application.


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit

**Project:** Student Academic Success Predictor  
**Task:** Multi-class Classification  
**Dataset Source:** UCI Machine Learning Repository
