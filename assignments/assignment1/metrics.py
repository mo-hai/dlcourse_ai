import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    
    TP = FP = TN = FN = 0
    for i in range(len(prediction)):
        if prediction[i] and ground_truth[i]:
            TP += 1
        elif prediction[i] and not ground_truth[i]:
            FP += 1
        elif not prediction[i] and ground_truth[i]:
            FN += 1
        else:
            TN += 1

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    accuracy = np.sum(prediction == ground_truth) / len(ground_truth)
    f1 = 2 * recall * precision / (recall + precision)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = np.sum(prediction == ground_truth) / len(ground_truth)
    return accuracy
