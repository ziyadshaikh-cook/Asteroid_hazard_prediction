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
    report_output_path: str = os.path.join("artifacts", "monitoring_report.html")


class ModelMonitoring:
    def __init__(self):
        self.config = ModelMonitoringConfig()

    def run_monitoring(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        try:
            logging.info("Building Evidently monitoring report")

            feature_cols = [
                "absolute_magnitude",
                "estimated_diameter_max",
                "relative_velocity",
                "miss_distance",
            ]

            reference_features = reference_data[feature_cols].dropna()
            current_features = current_data[feature_cols].dropna()

            logging.info(f"Reference shape: {reference_features.shape}")
            logging.info(f"Current shape: {current_features.shape}")

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

            logging.info(f"Report saved to: {self.config.report_output_path}")
            print(f"\nReport saved: {self.config.report_output_path}")

        except Exception as e:
            raise CustomException(e, sys)