# Asteroid Hazard Prediction

An end-to-end machine learning project that predicts whether a near-Earth asteroid is potentially hazardous to Earth, using real orbital and physical measurements from NASA's live NeoWs API.

---

## Live Demo

Enter a date range → the app fetches real asteroids from NASA's database → the ML model predicts which ones are potentially hazardous.

---

## Project Overview

| Item | Detail |
|---|---|
| Problem Type | Binary Classification |
| Target | `is_hazardous` — True / False |
| Dataset | NASA Nearest Earth Objects 1910–2024 (~338,000 records) |
| Best Model | Random Forest (F1: 0.66, Recall: 0.72, Precision: 0.61) |
| Data Source (Training) | Kaggle — NASA Nearest Earth Objects |
| Data Source (Inference) | NASA NeoWs REST API (live) |

---

## What Makes This Project Different

Most ML projects train and predict on the same static dataset. This project separates training from inference:

- **Training:** Historical Kaggle CSV (338,000 asteroid records, 1910–2024)
- **Inference:** Live NASA NeoWs API — the Flask app fetches real asteroids for any date range the user enters and runs predictions on fresh data in real time

---

## Project Architecture
User enters date range
↓
Flask app calls NASA NeoWs API
↓
JSON response parsed → DataFrame
↓
Preprocessor (RobustScaler) transforms features
↓
Random Forest model predicts hazard status
↓
Results displayed with hazard probability scores

---

## ML Pipeline

### Problem
Predict whether an asteroid is potentially hazardous based on 4 features:
- `absolute_magnitude` — brightness (lower = larger)
- `estimated_diameter_max` — maximum estimated size in km
- `relative_velocity` — speed relative to Earth in km/h
- `miss_distance` — closest approach distance in km

### Class Imbalance
87.2% not hazardous / 12.8% hazardous — handled with SMOTE on train set only.

### Why Accuracy is Not Used
A model that always predicts "not hazardous" gets 87% accuracy. That model is useless. Primary metric is F1-score and Recall on the hazardous class.

### Model Selection

| Model | F1 (Hazardous) | Recall | Precision |
|---|---|---|---|
| Logistic Regression | 0.4583 | 0.8773 | 0.3101 |
| **Random Forest** | **0.6606** | **0.7194** | **0.6107** |
| Gradient Boosting | 0.4841 | 0.9789 | 0.3216 |
| XGBoost | 0.4878 | 0.9870 | 0.3239 |
| CatBoost | 0.5000 | 0.9588 | 0.3382 |

Random Forest was selected — only model with Precision above 0.60, meaning it actually discriminates between classes rather than flagging everything as hazardous.

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML | Scikit-learn, XGBoost, CatBoost |
| Data | Pandas, NumPy |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Experiment Tracking | MLflow + DagsHub |
| API Integration | requests (NASA NeoWs REST API) |
| Monitoring | Evidently |
| Web App | Flask |
| Containerization | Docker |

---

## Project Structure
asteroid_hazard_prediction/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # CSV loader + NASA API fetch
│   │   ├── data_transformation.py  # RobustScaler + SMOTE
│   │   ├── model_trainer.py        # Random Forest + MLflow logging
│   │   └── model_monitoring.py     # Evidently drift report
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Chains ingestion → transform → train
│   │   ├── prediction_pipeline.py  # Loads model + preprocessor for inference
│   │   └── monitoring_pipeline.py  # Runs drift detection
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
├── notebook/
│   ├── 01_EDA.ipynb
│   └── 02_model_experiment.ipynb
├── templates/
│   ├── index.html
│   └── results.html
├── app.py                          # Flask application
├── main.py                         # Runs full training pipeline
├── Dockerfile
└── requirements.txt

---

## Setup and Usage

### 1. Clone the repository
```bash
git clone https://github.com/ziyadshaikh-cook/Asteroid_hazard_prediction.git
cd Asteroid_hazard_prediction
```

### 2. Create environment
```bash
conda create -p venv python=3.10 -y
conda activate ./venv
pip install -r requirements.txt
pip install -e .
```

### 3. Set up environment variables
Create a `.env` file at the root:
NASA_API_KEY=your_nasa_api_key
DAGSHUB_TOKEN=your_dagshub_token
Get a free NASA API key at: https://api.nasa.gov

### 4. Download the dataset
Download from Kaggle: [NASA Nearest Earth Objects 1910-2024](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024)
Place it as `data/raw.csv`

### 5. Run the training pipeline
```bash
python main.py
```

### 6. Run the Flask app
```bash
python app.py
```
Open `http://localhost:5000`

---

## Docker

```bash
docker build -t asteroid-hazard .
docker run -e NASA_API_KEY=your_key -e DAGSHUB_TOKEN=your_token -p 5000:5000 asteroid-hazard
```

---

## Experiment Tracking

All model runs are logged to DagsHub:
https://dagshub.com/ziyadshaikh-cook/Asteroid_hazard_prediction.mlflow

---

## Key Design Decisions

**Why RobustScaler over StandardScaler?**
Diameter and velocity features are heavily right-skewed with outliers. RobustScaler uses median and IQR instead of mean and std, making it more resistant to outliers.

**Why SMOTE after scaling?**
SMOTE creates synthetic points by interpolating between existing ones. Interpolating in the unscaled space produces synthetic points in the wrong distribution. Scale first, then SMOTE.

**Why not use the live API for training?**
The NASA API returns a maximum 7-day window per call. Collecting 338,000 records would require hundreds of API calls over hours. The Kaggle CSV is the same data pre-collected — use it for training, use the live API for inference.
