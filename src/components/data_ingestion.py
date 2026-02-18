import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("entered the data ingestion method or component")
        try:
            df = pd.read_csv(r"D:\Desktop/gmail-api-automation/artifacts/training_emails.csv")
            df2 = pd.read_csv(r"D:\Desktop/gmail-api-automation/artifacts/test_labels_for_evaluation.csv")
            # Select only IMPORTANT rows from df2
            #important_df2 = df2[df2['label'] == 'IMPORTANT']

            # Append rows
            df = pd.concat([df, df2], ignore_index=True)
            df = df[["text", "label"]]

            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(
                df, 
                test_size=0.25, 
                random_state=39,
                stratify=df["label"]
                )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    print("RUNNING:", __file__)
    logging.info("Starting data ingestion")


    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        logging.info(f"Train data saved at: {train_data}")
        logging.info(f"Test data saved at: {test_data}")

        data_transformation = DataTransformation()
        X_train_transformed,y_train_encoded,X_test_transformed,y_test_encoded,_,_ = data_transformation.initiate_data_transformation(train_data, test_data)
        modelTrainer = ModelTrainer()
        print(modelTrainer.initiate_model_trainer(X_train_transformed,y_train_encoded,X_test_transformed,y_test_encoded))

    except Exception as e:
        logging.error("Error occurred in data ingestion")
        raise CustomException(e, sys)
