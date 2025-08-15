import numpy as np

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy for binary classification.

    Balanced accuracy = (Sensitivity + Specificity) / 2

    Where:
    - Sensitivity (Recall) = TP / (TP + FN) = True Positive Rate
    - Specificity = TN / (TN + FP) = True Negative Rate

    :param y_true: True binary labels (0 or 1)
    :param y_pred :Predicted binary labels (0 or 1)
    :return Balanced accuracy score between 0 and 1
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Balanced accuracy is the average
    return (sensitivity + specificity) / 2

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall (sensitivity) for binary classification.

    Recall = TP / (TP + FN)
    :param y_true: True binary labels (0 or 1)
    :param y_pred: Predicted binary labels (0 or 1)
    :return: Recall score between 0 and 1
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives

    return tp / (tp + fn) if (tp + fn) > 0 else 0

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision for binary classification.

    Precision = TP / (TP + FP)
    :param y_true: True binary labels (0 or 1)
    :param y_pred: Predicted binary labels (0 or 1)
    :return: Precision score between 0 and 1
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives

    return tp / (tp + fp) if (tp + fp) > 0 else 0

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score for binary classification.

    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    :param y_true: True binary labels (0 or 1)
    :param y_pred: Predicted binary labels (0 or 1)
    :return: F1 score between 0 and 1
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def mcc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Matthews correlation coefficient (MCC) for binary classification.

    MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    :param y_true: True binary labels (0 or 1)
    :param y_pred: Predicted binary labels (0 or 1)
    :return: MCC score between -1 and 1
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / denominator if denominator > 0 else 0