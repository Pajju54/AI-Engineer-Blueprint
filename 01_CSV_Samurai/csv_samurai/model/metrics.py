from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import os
from logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model_pipeline, X_val, y_val, save_dir="csv_samurai/artifacts"):
    """
    Evaluate the model using common classification metrics.

    Parameters:
    - model_pipeline: The trained pipeline (preprocessing + classifier).
    - X_val: Validation features.
    - y_val: True labels for validation data.

    Returns:
    - dict: A dictionary containing evaluation metrics.
    """
    logger.info("Evaluating the model performance on validation set...")

    y_pred = model_pipeline.predict(X_val)

    accuracy = accuracy_score(y_val,y_pred)
    precision = precision_score(y_val,y_pred, average='binary')
    recall = recall_score(y_val,y_pred, average='binary')
    f1 = f1_score(y_val,y_pred, average='binary')
    report = classification_report(y_val, y_pred)
    logger.info("Classification Report:\n" + report)


    cm = confusion_matrix(y_val,y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {cm_path}")

    #Wrap it up
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        "confusion_matrix":cm.tolist(), # For serialization/logging
    }
    return metrics