import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Building preprocessing pipeline")

            pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ])

            return pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

            # Drop columns decided in EDA
            cols_to_drop = ["neo_id", "name", "orbiting_body", "estimated_diameter_min"]
            train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
            test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])

            target_column = "is_hazardous"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column].astype(int)

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column].astype(int)

            # Build and fit preprocessor on train only
            preprocessor = self.get_data_transformer_object()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            logging.info("Preprocessing complete. Applying SMOTE on train set only.")

            # SMOTE on train only — never on test
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

            logging.info(
                f"After SMOTE — Train shape: {X_train_resampled.shape} | "
                f"Class distribution: {np.bincount(y_train_resampled)}"
            )

            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            logging.info("Preprocessor saved to artifacts/preprocessor.pkl")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)