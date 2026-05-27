import os
import sys
import pickle
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import f1_score, recall_score, precision_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report[name] = {
                "f1": round(f1_score(y_test, y_pred), 4),
                "recall": round(recall_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred), 4),
            }
            logging.info(f"{name} — F1: {report[name]['f1']} | Recall: {report[name]['recall']}")
        return report
    except Exception as e:
        raise CustomException(e, sys)