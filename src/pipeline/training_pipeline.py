import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # Step 1 — Data Ingestion
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Ingestion complete. Train: {train_path} | Test: {test_path}")

            # Step 2 — Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path, test_path
            )
            logging.info(f"Transformation complete. Preprocessor saved at: {preprocessor_path}")

            # Step 3 — Model Training
            model_trainer = ModelTrainer()
            f1 = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Training complete. Final F1: {f1}")

            return f1

        except Exception as e:
            raise CustomException(e, sys)