import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import CleanTextTransformer, get_vectorizer, save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )
    label_encoder_file_path: str = os.path.join(
        "artifacts", "label_encoder.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            text_pipeline = Pipeline(
                    steps=[
                        ("clean_text", CleanTextTransformer()),
                        ("tfidf", get_vectorizer()),
                    ]
)


            preprocessor = ColumnTransformer(
                transformers=[
                    ("text_pipeline", text_pipeline, ["text"])
                ]
            )

            logging.info("Text cleaning and vectorization pipeline created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # --------------------------
            # Read Data
            # --------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print("Train DF shape:", train_df.shape)
            print(train_df.columns)
            #print(train_df.shape)
            print(train_df.head())



            logging.info("Train and test data loaded successfully")

            target_column_name = "label"

            # --------------------------
            # Split Features and Target
            # --------------------------
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            # --------------------------
            # Preprocessing Object
            # --------------------------
            preprocessor_obj = self.get_data_transformer_object()

            logging.info("Fitting preprocessor on training data only")

            # Fit only on TRAIN → prevents leakage
            X_train_transformed = preprocessor_obj.fit_transform(X_train)

            # Only transform TEST
            X_test_transformed = preprocessor_obj.transform(X_test)

            
            print("X_train_transformed",X_train_transformed.shape)
            print("X_test_transformed",X_test_transformed.shape)
            print(type(X_train_transformed))


            # --------------------------
            # Label Encoding
            # --------------------------
            label_encoder = LabelEncoder()

            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            print(y_train_encoded.shape)

            # --------------------------
            # Combine Features + Target
            # --------------------------
            X_train_transformed = X_train_transformed.toarray()
            X_test_transformed = X_test_transformed.toarray()

            y_train_encoded = y_train_encoded.reshape(-1, 1)
            y_test_encoded = y_test_encoded.reshape(-1, 1)

            train_arr = np.hstack((X_train_transformed, y_train_encoded))
            test_arr = np.hstack((X_test_transformed, y_test_encoded))



            # --------------------------
            # Save Objects
            # --------------------------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            save_object(
                file_path=self.data_transformation_config.label_encoder_file_path,
                obj=label_encoder,
            )

            logging.info("Preprocessor and label encoder saved successfully")
            print(y_train_encoded.shape)
            print(type(y_train_encoded))


            return (
                
                X_train_transformed,
                y_train_encoded,
                X_test_transformed,
                y_test_encoded,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_file_path,

            )

        except Exception as e:
            raise CustomException(e, sys)

