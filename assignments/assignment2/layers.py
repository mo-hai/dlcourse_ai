import numpy as np
import math


def softmax(predictions: np.array) -> np.array:
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    """

    pred = predictions.copy()

    x = np.exp(pred - np.max(pred, axis=1).reshape(pred.shape[0], -1)) # to avoid numerical instability
    res = x / np.sum(x, axis=1).reshape(pred.shape[0], -1)
    
    assert res.shape == predictions.shape

    return res


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    """
    # minus sign is added to make it a loss function, so we minimize it
    return - np.mean(np.log(probs[range(target_index.shape[0]), target_index.flatten()])) # can be sum instead of mean


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    W_copy = W.copy()

    loss = reg_strength * np.sum(W_copy ** 2)
    grad = 2 * reg_strength * W_copy

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)

    # gradient of predictions by loss value
    # dL/dw_ji = h_j * (y_pred_i - gt_i)
    indxs = np.zeros(probs.shape)
    indxs[[range(len(target_index))], target_index.flatten()] = 1
    dpreds = probs - indxs

    assert dpreds.shape == preds.shape

    dpreds /= dpreds.shape[0]

    return loss, dpreds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.relu_grad = None

    def forward(self, X: np.array) -> np.array:
        result = np.maximum(X, 0)
        
        self.relu_grad = result.copy()
        self.relu_grad[self.relu_grad > 0] = 1
        
        assert self.relu_grad.shape == X.shape
        assert X.max() == result.max()

        return result

    def backward(self, d_out: np.array) -> np.array:
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """

        d_result = self.relu_grad * d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input: int, n_output: int):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        result = X @ self.params()['W'].value + self.params()['B'].value
        return result

    def backward(self, d_out: np.array) -> np.array:
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        self.params()['W'].grad += self.X.T @ d_out # dW
        self.params()['B'].grad += np.sum(d_out, axis=0).reshape(1, -1) # dB
        d_input = d_out @ self.params()['W'].value.T # dX

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
