import math
import numpy as np
from typing import Union


def softmax(predictions: np.array) -> np.array:
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''

    pred = predictions.copy()
    if pred.ndim > 1:
      x = np.exp(pred - np.max(pred, axis=1).reshape(pred.shape[0], -1))
      res = x / np.sum(x, axis=1).reshape(pred.shape[0], -1)
    else:
      x = np.exp(pred - np.max(pred))
      res = x / np.sum(x)
    
    assert res.shape == predictions.shape

    return res


def cross_entropy_loss(probs: np.array, target_index: Union[np.array, int]) -> float:
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    if isinstance(target_index, int):
      return - math.log(probs[target_index])
    
    return - np.mean(np.log(probs[range(target_index.shape[0]), target_index.flatten()])) # can be sum instead of mean


def softmax_with_cross_entropy(predictions: np.array, target_index: Union[np.array, int]) -> tuple[float, np.array]:
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''

    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    # gradient - derivative of loss function
    # dL/dw_ji = h_j * (y_pred_i - gt_i)
    if isinstance(target_index, int):
      indxs = np.array([int(i==target_index) for i in range(len(probs))])
    else:
      indxs = np.zeros(probs.shape)
      indxs[[range(len(target_index))], target_index.flatten()] = 1

    dprediction = probs - indxs
    assert dprediction.shape == predictions.shape

    if isinstance(target_index, type(np.array([]))):
      dprediction /= dprediction.shape[0] # if sum is used in cross_entropy_loss than this should be commented

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    
    loss = np.sum(reg_strength * W ** 2)
    grad = reg_strength * W * 2 # derivative of L2 regularization loss -> reg_strength * W ** 2

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, grad = softmax_with_cross_entropy(predictions, target_index)

    dW = X.T @ grad # update weights using gradient
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self, verbose=False):
        self.W = None
        self.verbose = verbose

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            batch_loss = 0
            for batch in batches_indices:
                batch_X = X[batch]
                batch_y = y[batch]

                loss, grad = linear_softmax(batch_X, self.W, batch_y)
                reg_loss, reg_grad = l2_regularization(self.W, reg)

                loss += reg_loss
                grad += reg_grad

                self.W -= learning_rate * grad
                batch_loss += loss
            
            loss = batch_loss / batch_size
            loss_history.append(loss)
            if self.verbose:
              print(f"Epoch {epoch}, loss: {loss}")

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        predictions = X @ self.W
        probs = softmax(predictions)
        y_pred = np.argmax(probs, axis=1) # get the index of the highest probability

        assert y_pred.shape[0] == X.shape[0]

        return y_pred
