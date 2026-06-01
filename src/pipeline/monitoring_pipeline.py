import sys
from src.exception import CustomException
from src.logger import logging
from src.components.model_monitoring import ModelMonitoring


class MonitoringPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            logging.info("========== Monitoring Pipeline Started ==========")
            monitor = ModelMonitoring()
            monitor.run_monitoring()
            logging.info("========== Monitoring Pipeline Completed ==========")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = MonitoringPipeline()
    pipeline.run()