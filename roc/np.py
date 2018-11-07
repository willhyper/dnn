from sklearn import metrics


def roc_curve(y_true, y_pred):
    return metrics.roc_curve(y_true, y_pred)
