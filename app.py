import os
import sys
import pandas as pd
from flask import Flask, request, render_template
from dotenv import load_dotenv

from src.pipeline.prediction_pipeline import PredictPipeline
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException

load_dotenv()

print("NASA KEY:", os.getenv("NASA_API_KEY"))

app = Flask(__name__)
NASA_API_KEY = os.getenv("NASA_API_KEY")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        logging.info(f"Fetching asteroids: {start_date} to {end_date}")

        # Fetch live data from NASA API
        ingestion = DataIngestion()
        df_live = ingestion.fetch_from_api(start_date, end_date, NASA_API_KEY)

        if df_live.empty:
            return render_template(
                "index.html",
                error="No asteroids found for this date range."
            )

        # Run predictions
        pipeline = PredictPipeline()
        predictions = pipeline.predict(df_live)
        probabilities = pipeline.predict_proba(df_live)

        # Build results
        df_live["prediction"] = ["HAZARDOUS" if p == 1 else "SAFE" for p in predictions]
        df_live["hazard_probability"] = [f"{p*100:.1f}%" for p in probabilities]
        df_live["absolute_magnitude"] = df_live["absolute_magnitude"].round(2)
        df_live["estimated_diameter_max"] = df_live["estimated_diameter_max"].round(4)
        df_live["relative_velocity"] = df_live["relative_velocity"].round(2)
        df_live["miss_distance"] = df_live["miss_distance"].round(0).astype(int)

        results = df_live.to_dict(orient="records")
        hazardous_count = sum(1 for r in results if r["prediction"] == "HAZARDOUS")
        safe_count = len(results) - hazardous_count

        logging.info(f"Predictions complete. Hazardous: {hazardous_count} | Safe: {safe_count}")

        return render_template(
            "results.html",
            results=results,
            start_date=start_date,
            end_date=end_date,
            total=len(results),
            hazardous_count=hazardous_count,
            safe_count=safe_count,
        )

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)