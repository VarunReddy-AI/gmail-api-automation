import re
import os
import sys
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

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



def model_metrics(X_train,X_test,y_train,y_test):
    model = joblib.load(r"D:\Desktop/gmail-api-automation/artifacts/model.pkl")

    y_pred = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy of test data:", accuracy_test)
    print("classification report: \n", classification_report(y_test, y_pred))

    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
    print("Accuracy of train data: ", accuracy_train) 
    print("classification report: \n", classification_report(y_train, y_pred_train))

    return accuracy_test, accuracy_train



