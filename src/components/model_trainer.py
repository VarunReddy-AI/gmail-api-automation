import os
import sys
from dataclasses import dataclass
from scipy.sparse import issparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, classification_report


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, evaluate_and_log_metrics


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        
        try:
            print(type(X_test))
            if issparse(X_train):
                X_train = X_train.toarray()

            if issparse(X_test):
                X_test = X_test.toarray()
            logging.info("Starting model training for classification")

            models = {
                "Logistic Regression": LogisticRegression(), #max_iter=2000)  this is giving me good accuracy
                #"Linear SVC": LinearSVC()   this model is overfitting
                #"Naive Bayes": MultinomialNB(),  this model is under performing
                #"Random Forest": RandomForestClassifier() this model is under performing
            }

            params = {
                "Logistic Regression": {
                        "C": [0.1, 0.05, 1, 10],
                        "max_iter": [2000],
                        "class_weight": [{
                            0: 8.0,
                            1: 0.76,
                            2: 1.28,
                            3: 0.41,
                            4: 2.83
                        }],
                        "n_jobs": [1]
                },
                "Linear SVC": {
                    "C": [0.1, 1, 10]
                },
                "Naive Bayes": {
                    "alpha": [0.1, 0.5, 1.0]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]
            print(model_report)
            if best_model_score < 0.70:
                raise CustomException("No good classification model found!", sys)

            logging.info(f"Best model found: {best_model_name}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            print(f"Best model: {best_model_name}")
            print("Best model score: ",best_model_score)
            accuracy_train, accuracy_test = evaluate_and_log_metrics(X_train,X_test,y_train,y_test,model_path="artifacts/model.pkl")
            
            logging.info(f"Accuracy of train: {accuracy_train}")
            logging.info(f"Accuracy of test: {accuracy_test}")
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
