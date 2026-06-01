import os
import sys
import pandas as pd
from dataclasses import dataclass

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelMonitoringConfig:
    reference_data_path: str = os.path.join("artifacts", "train.csv")
    current_data_path: str = os.path.join("artifacts", "test.csv")
    report_output_path: str = os.path.join("artifacts", "monitoring_report.html")


class ModelMonitoring:
    def __init__(self):
        self.config = ModelMonitoringConfig()

    def run_monitoring(self):
        try:
            logging.info("Loading reference and current datasets for monitoring")

            reference_data = pd.read_csv(self.config.reference_data_path)
            current_data = pd.read_csv(self.config.current_data_path)

            logging.info(f"Reference data shape: {reference_data.shape}")
            logging.info(f"Current data shape: {current_data.shape}")

            feature_cols = [
                "absolute_magnitude",
                "estimated_diameter_max",
                "relative_velocity",
                "miss_distance",
            ]

            reference_features = reference_data[feature_cols].dropna()
            current_features = current_data[feature_cols].dropna()

            logging.info("Building Evidently monitoring report")

            report = Report([
                DataDriftPreset(),
                DataSummaryPreset(),
            ])

            my_report = report.run(
                reference_data=reference_features,
                current_data=current_features
            )

            os.makedirs(os.path.dirname(self.config.report_output_path), exist_ok=True)
            my_report.save_html(self.config.report_output_path)

            logging.info(f"Monitoring report saved to: {self.config.report_output_path}")
            print(f"\nReport saved: {self.config.report_output_path}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    monitor = ModelMonitoring()
    monitor.run_monitoring()