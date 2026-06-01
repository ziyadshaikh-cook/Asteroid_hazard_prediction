import os
import sys
import pandas as pd
from dotenv import load_dotenv

from src.exception import CustomException
from src.logger import logging
from src.components.model_monitoring import ModelMonitoring
from src.components.data_ingestion import DataIngestion

load_dotenv()


class MonitoringPipeline:
    def __init__(self):
        pass

    def run(self, start_date: str, end_date: str):
        try:
            logging.info("========== Monitoring Pipeline Started ==========")

            # Reference = training data
            reference_data = pd.read_csv("artifacts/train.csv")
            logging.info(f"Reference data loaded: {reference_data.shape}")

            # Current = live NASA API data
            api_key = os.getenv("NASA_API_KEY")
            ingestion = DataIngestion()
            current_data = ingestion.fetch_from_api(start_date, end_date, api_key)
            logging.info(f"Live data fetched: {current_data.shape}")

            # Run monitoring
            monitor = ModelMonitoring()
            monitor.run_monitoring(
                reference_data=reference_data,
                current_data=current_data
            )

            logging.info("========== Monitoring Pipeline Completed ==========")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = MonitoringPipeline()
    pipeline.run("2021-01-01", "2021-01-07")