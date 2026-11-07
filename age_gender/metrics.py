import numpy as np


def calculate(true_age, pred_age, true_gender, pred_gender):
    a, p = np.asarray(true_age), np.asarray(pred_age)
    g, q = np.asarray(true_gender), np.asarray(pred_gender)
    if not len(a) or not (len(a) == len(p) == len(g) == len(q)):
        raise ValueError('Metric arrays must be nonempty and have matching lengths')
    cm = [[int(((g == i) & (q == j)).sum()) for j in range(2)] for i in range(2)]
    classes = []
    for i in range(2):
        tp = cm[i][i]
        precision = tp / max(1, sum(row[i] for row in cm))
        recall = tp / max(1, sum(cm[i]))
        classes.append({'label': i, 'precision': precision, 'recall': recall,
                        'f1': 2 * precision * recall / max(1e-12, precision + recall),
                        'support': sum(cm[i])})
    return {'samples': len(a), 'age_mae_years': float(np.abs(a - p).mean()),
            'age_rmse_years': float(np.sqrt(np.square(a - p).mean())),
            'gender_accuracy': float((g == q).mean()), 'confusion_matrix': cm,
            'classes': classes}
