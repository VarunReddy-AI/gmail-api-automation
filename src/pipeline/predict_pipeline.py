import os
import joblib
import pandas as pd


class EmailClassifier:

    def __init__(self, artifact_path="artifacts"):
        self.artifact_path = artifact_path

        self.model = joblib.load(
            os.path.join(self.artifact_path, "model.pkl")
        )

        self.preprocessor = joblib.load(
            os.path.join(self.artifact_path, "preprocessor.pkl")
        )

        self.label_encoder = joblib.load(
            os.path.join(self.artifact_path, "lable_encoder.pkl")
        )

    def predict(self, text: str) -> str:
        df = pd.DataFrame({"text": [text]})

        X = self.preprocessor.transform(df)
        prediction = self.model.predict(X)

        label = self.label_encoder.inverse_transform(prediction)

        return label[0]