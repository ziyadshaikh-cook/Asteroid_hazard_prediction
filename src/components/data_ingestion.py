import os
import sys
import pandas as pd
import requests
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df = pd.read_csv("data/raw.csv")
            logging.info(f"Dataset loaded: {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["is_hazardous"]
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(f"Train size: {train_set.shape} | Test size: {test_set.shape}")
            logging.info("Data ingestion complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def fetch_from_api(self, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
        """
        Fetches live asteroid data from NASA NeoWs API.
        Date format: YYYY-MM-DD
        Max date range per call: 7 days (NASA API limit).
        Returns a DataFrame with the same columns as the training data.
        """
        logging.info(f"Fetching live data from NASA API: {start_date} to {end_date}")
        try:
            url = "https://api.nasa.gov/neo/rest/v1/feed"
            params = {
                "start_date": start_date,
                "end_date": end_date,
                "api_key": api_key,
            }

            response = requests.get(url, params=params)

            if response.status_code != 200:
                raise CustomException(
                    f"NASA API returned status {response.status_code}: {response.text}", sys
                )

            data = response.json()
            neo_list = data.get("near_earth_objects", {})

            records = []
            for date, asteroids in neo_list.items():
                for asteroid in asteroids:
                    try:
                        diameter = asteroid["estimated_diameter"]["kilometers"]
                        close_approach = asteroid["close_approach_data"][0]

                        record = {
                            "name": asteroid.get("name", "Unknown"),
                            "absolute_magnitude": float(asteroid.get("absolute_magnitude_h", 0)),
                            "estimated_diameter_max": float(diameter["estimated_diameter_max"]),
                            "relative_velocity": float(
                                close_approach["relative_velocity"]["kilometers_per_hour"]
                            ),
                            "miss_distance": float(
                                close_approach["miss_distance"]["kilometers"]
                            ),
                            "is_hazardous": bool(
                                asteroid.get("is_potentially_hazardous_asteroid", False)
                            ),
                        }
                        records.append(record)
                    except (KeyError, IndexError):
                        continue

            df_live = pd.DataFrame(records)
            logging.info(f"Live API fetch complete: {df_live.shape[0]} asteroids retrieved")
            return df_live

        except Exception as e:
            raise CustomException(e, sys)