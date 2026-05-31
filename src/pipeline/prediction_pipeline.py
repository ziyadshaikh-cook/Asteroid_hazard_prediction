import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features: pd.DataFrame) -> pd.Series:
        try:
            logging.info("Loading model and preprocessor")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Expected columns in this exact order
            expected_cols = [
                "absolute_magnitude",
                "estimated_diameter_max",
                "relative_velocity",
                "miss_distance",
            ]

            # Ensure correct columns and order
            features = features[expected_cols]

            logging.info(f"Input shape: {features.shape}")
            scaled = preprocessor.transform(features)
            predictions = model.predict(scaled)
            logging.info("Prediction complete")

            return predictions

        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_proba(self, features: pd.DataFrame):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            expected_cols = [
                "absolute_magnitude",
                "estimated_diameter_max",
                "relative_velocity",
                "miss_distance",
            ]
            features = features[expected_cols]
            scaled = preprocessor.transform(features)
            proba = model.predict_proba(scaled)[:, 1]  # probability of class 1 (hazardous)
            return proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Used when a single asteroid's data is passed manually.
    Converts it into a DataFrame the predict pipeline can consume.
    """
    def __init__(
        self,
        absolute_magnitude: float,
        estimated_diameter_max: float,
        relative_velocity: float,
        miss_distance: float,
    ):
        self.absolute_magnitude = absolute_magnitude
        self.estimated_diameter_max = estimated_diameter_max
        self.relative_velocity = relative_velocity
        self.miss_distance = miss_distance

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "absolute_magnitude": [self.absolute_magnitude],
                "estimated_diameter_max": [self.estimated_diameter_max],
                "relative_velocity": [self.relative_velocity],
                "miss_distance": [self.miss_distance],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)