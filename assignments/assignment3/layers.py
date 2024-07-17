import numpy as np


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

    preds = np.exp(predictions - np.max(predictions, axis=1).reshape(predictions.shape[0], -1))
    probs = preds / np.sum(preds, axis=1).reshape(predictions.shape[0], -1)

    return probs

def cross_entropy_loss(probs: np.array, target_index: np.array) -> float:
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
    '''
    loss = -np.mean(np.log(probs[np.arange(target_index.shape[0]), target_index.flatten()]))
    return loss

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
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
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

    indexs = np.arange(target_index.shape[0])
    probs[indexs, target_index.flatten()] -= 1
    dprediction = probs / target_index.shape[0]
    
    assert dprediction.shape == predictions.shape

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.relu = None

    def forward(self, X):
        result = np.maximum(X, 0)
        assert X.max() == result.max()

        self.relu = result.copy()
        self.relu[self.relu > 0] = 1
        assert self.relu.shape == X.shape

        return result

    def backward(self, d_out):
        d_result = d_out * self.relu
        return d_result

    def params(self):
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
        self.params()['W'].grad += self.X.T @ d_out # dW
        self.params()['B'].grad += np.sum(d_out, axis=0).reshape(1, -1) # dB
        d_input = d_out @ self.params()['W'].value.T # dX

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.W = Param(np.random.randn(filter_size, filter_size, in_channels, out_channels))
        self.B = Param(np.zeros(out_channels))


    def forward(self, X):
        batch_size, height, width, _ = X.shape

        self.X = X.copy()
        self.X_pad = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)

        out_height = (height + 2 * self.padding - self.filter_size) + 1
        out_width = (width + 2 * self.padding - self.filter_size) + 1
        
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                X_slice = self.X_pad[:, y:y+self.filter_size, x:x+self.filter_size, :]
                result[:, y, x, :] = np.sum(X_slice[:, :, :, :, np.newaxis] * self.params()['W'].value[np.newaxis, :, :, :, :], axis=(1, 2, 3))
        
        assert X.shape == (batch_size, height, width, self.in_channels)
        assert result.shape == (batch_size, out_height, out_width, self.out_channels)
        
        return result + self.params()['B'].value


    def backward(self, d_out):

        _, height, width, _ = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        d_input = np.zeros(self.X_pad.shape) # dL/dX

        for y in range(out_height):
            for x in range(out_width):
                dout_slice = d_out[:, y:y+1, x:x+1, np.newaxis, :]

                X_slice = self.X_pad[:, y:y+self.filter_size, x:x+self.filter_size, :, np.newaxis]
                self.params()['W'].grad += np.sum(X_slice * dout_slice, axis=0)

                d_input[:, y:y+self.filter_size, x:x+self.filter_size, :] += np.sum(self.params()['W'].value[np.newaxis, :, :, :, :] * dout_slice, axis=4)
                
                dout_slice = d_out[:, y:y+1, x:x+1, :]
                self.params()['B'].grad += np.sum(dout_slice, axis=(0, 1, 2))
                
        return d_input[:, self.padding:height+self.padding, self.padding:width+self.padding, :]


    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = X.copy()
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        result = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(0, out_height):
            for x in range(0, out_width):
                X_slice = X[:, y*self.stride:y*self.stride+self.pool_size, x*self.stride:x*self.stride+self.pool_size, :]
                result[:, x, y, :] = np.max(X_slice, axis=(1, 2))

        return result

    def backward(self, d_out):

        batch_size, in_height, in_width, in_channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        d_result = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):

                pool_y_from = y * self.stride
                pool_y_to = pool_y_from + self.pool_size
                pool_x_from = x * self.stride
                pool_x_to = pool_x_from + self.pool_size

                for b in range(batch_size):
                    for c in range(in_channels):
                        d_out_pooled = d_out[b, y, x, c]
                        X_pooled = self.X[b, pool_y_from:pool_y_to, pool_x_from:pool_x_to, c]

                        max_ind_y, max_ind_x = np.unravel_index(np.argmax(X_pooled), X_pooled.shape)
                        d_result[b, pool_y_from + max_ind_y, pool_x_from + max_ind_x, c] += d_out_pooled


        return d_result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
