import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from load import read_config
from logger import get_logger
from features.pipeline import build_preprocessing_pipeline 
from model.metrics import evaluate_model
import json
import os

logger = get_logger(__name__)

def train_and_evaluate(data: pd.DataFrame, target_column: str):
    config = read_config()
    model_config = config["model"]

    logger.info("Splitting dataset into train and validation")
    X = data.drop(columns=target_column, axis=1)
    y = data[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=model_config["val_split"],random_state=model_config["random_state"], stratify=y)
    logger.info("Bulding preprocessing pipeline...")
    preprocessing = build_preprocessing_pipeline()

    logger.info("Creating full pipeline with the model...")
    clf = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        random_state=model_config["random_state"]
    )

    pipeline = Pipeline(steps=[
        ('preprocessing',preprocessing),
        ("classifier", clf)
    ])

    logger.info("Training the model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model on Validation data...")

    metrics = evaluate_model(pipeline, X_val,y_val)

    logger.info(f"Validation metrics: {metrics}")

    metrics_path = config["artifacts"]["metrics_path"]
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

    # Save the model
    joblib.dump(pipeline, config["artifacts"]["model_path"])
    logger.info(f"Model saved to {config['artifacts']['model_path']}")



    return metrics