import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

        def get_data_transform_object(self):
            try:
                
                text_pipeline = Pipeline(
                    steps=[
                        ("clean_text", clean_text()),
                        ("tfidf", TfidfVectorizer(
                            max_features=10000,
                            ngram_range=(1,2),
                            min_df=2,
                            max_df=0.9,
                            sublinear_tf=True
                        ))
                    ]
                )

                logging.info("Cleaning text and vectoring")
                return text_pipeline

            except Exception as e:
                raise CustomException(e,sys)
            

def initiate_data_transformation(self, train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(train_path)

        logging.info("Read train and test data completed")

        logging.info("Obtaining preprocessing object")

        preprocessor_obj = self.get_data_transformer_object()

        target_column_name = "label"
        numerical_columns = "text"

        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
        target_feature_test_df = test_df[target_column_name]

        logging.info(f"applying preprocessing object in training and test dataframe")

        input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

        train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df)
        ]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        logging.info("saving preprocessing object")

        save_object(
            file_path = self.data_transformation_config.preprocessor_obj_file_path,
            obj = preprocessor_obj
        )

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,
        )
    except Exception as e:
        raise CustomException(e, sys)