import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes TP, FP, TN, FN.
    Positive Class = 1 (SAFE)
    Negative Class = 0 (UNSAFE)
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # Catastrophic: Unsafe marked Safe
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # Conservative: Safe marked Unsafe
    
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = compute_confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    false_safe_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "False_Safe_Rate": false_safe_rate,
        "Confusion_Matrix": cm
    }


def print_report(metrics: dict):
    cm = metrics["Confusion_Matrix"]
    print("=" * 55)
    print("           PERCEPTRON EVALUATION REPORT")
    print("=" * 55)
    print(f"  Accuracy:          {metrics['Accuracy'] * 100:.2f}%")
    print(f"  Precision:         {metrics['Precision'] * 100:.2f}%")
    print(f"  Recall:            {metrics['Recall'] * 100:.2f}%")
    print(f"  F1 Score:          {metrics['F1_Score']:.4f}")
    print(f"  False-Safe Rate:   {metrics['False_Safe_Rate'] * 100:.2f}% (FP / Total Unsafe)")
    print("-" * 55)
    print("  Confusion Matrix:")
    print(f"    True SAFE correctly predicted (TP):   {cm['TP']}")
    print(f"    True UNSAFE correctly rejected (TN):  {cm['TN']}")
    print(f"    Safe spots missed (FN):               {cm['FN']}")
    print(f"    UNSAFE SPOTS MARKED SAFE (FP - DANGER): {cm['FP']}")
    print("=" * 55)