import re
import os
import sys
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import numpy as np 
import joblib
import pandas as pd
import pickle
import dill
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin

class CleanTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # ColumnTransformer passes a DataFrame with 1 column
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        return X.apply(self._clean)

    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)

        if '<' in text and '>' in text:
            text = BeautifulSoup(text, 'html.parser').get_text()

        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

"""def clean_text(text):
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    if '<' in text and '>' in text:
        text = BeautifulSoup(text, 'html.parser').get_text()

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

"""
def get_vectorizer():
    return TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True
    )



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_and_log_metrics(X_train, X_test, y_train, y_test, model_path):
    
    # Load model
    model = joblib.load(model_path)

    # Ensure correct shape
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # ---- Test Metrics ----
    y_pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)

    # ---- Train Metrics ----
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    train_report = classification_report(y_train, y_pred_train)

    # Print to console
    print("Test Confusion Matrix:\n", test_conf_matrix)
    print("Test Accuracy:", accuracy_test)
    print("Test Classification Report:\n", test_report)

    print("Train Confusion Matrix:\n", train_conf_matrix)
    print("Train Accuracy:", accuracy_train)
    print("Train Classification Report:\n", train_report)

    # ---- Save to Markdown ----
    metrics_path = r"D:\Desktop/gmail-api-automation/metrics.md"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("# Model Training Metrics\n\n")
        f.write(f"**Run Time:** {datetime.now()}\n\n")
        f.write(f"**Model Path:** {model_path}\n\n")

        f.write("## Test Metrics\n")
        f.write(f"- Accuracy: {accuracy_test:.4f}\n\n")
        f.write("### Confusion Matrix\n")
        f.write("```\n")
        f.write(str(test_conf_matrix))
        f.write("\n```\n\n")

        f.write("### Classification Report\n")
        f.write("```\n")
        f.write(test_report)
        f.write("\n```\n\n")

        f.write("## Train Metrics\n")
        f.write(f"- Accuracy: {accuracy_train:.4f}\n\n")
        f.write("### Confusion Matrix\n")
        f.write("```\n")
        f.write(str(train_conf_matrix))
        f.write("\n```\n\n")

        f.write("### Classification Report\n")
        f.write("```\n")
        f.write(train_report)
        f.write("\n```\n")

    return accuracy_test, accuracy_train
