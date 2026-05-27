import os
from pathlib import Path

folders = [
    "data",
    "notebook",
    "src/components",
    "src/pipeline",
    "templates",
    "artifacts",
]

files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_monitoring.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/pipeline/monitoring_pipeline.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",
    "notebook/01_EDA.ipynb",
    "notebook/02_model_experiment.ipynb",
    "templates/index.html",
    "templates/results.html",
    "app.py",
    "main.py",
    "requirements.txt",
    "setup.py",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

for file in files:
    path = Path(file)
    if not path.exists():
        path.touch()
        print(f"Created: {file}")
    else:
        print(f"Already exists: {file}")

print("\nScaffolding complete.")