import os
import sys
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

load_dotenv()


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            dagshub.init(
                repo_owner="ziyadshaikh-cook",
                repo_name="Asteroid_hazard_prediction",
                mlflow=True,
            )

            mlflow.set_experiment("Asteroid Hazard Prediction")

            model = RandomForestClassifier(
                n_estimators=10,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )
            logging.info("Training Random Forest")

            with mlflow.start_run(run_name="Random Forest"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                f1        = round(f1_score(y_test, y_pred), 4)
                recall    = round(recall_score(y_test, y_pred), 4)
                precision = round(precision_score(y_test, y_pred), 4)

                mlflow.log_param("model", "RandomForestClassifier")
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("class_weight", "balanced")
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("precision", precision)
                

                logging.info(f"F1: {f1} | Recall: {recall} | Precision: {precision}")

            if f1 < 0.6:
                raise CustomException("Model F1 below 0.6 threshold. Pipeline halted.", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )

            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            print(f"\nF1: {f1} | Recall: {recall} | Precision: {precision}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=["Not Hazardous", "Hazardous"]))

            return f1

        except Exception as e:
            raise CustomException(e, sys)