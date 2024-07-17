import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        height, width, channels = input_shape

        pool1 = stride1 = 4
        pool2 = stride2 = 4

        flattened_height = height // (pool1 * pool2)
        flattened_width = width // (pool1 * pool2)

        self.Conv1 = ConvolutionalLayer(channels, conv1_channels, 3, 1)
        self.Relu1 = ReLULayer()
        self.Pool1 = MaxPoolingLayer(pool1, stride1)
        self.Conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.Relu2 = ReLULayer()
        self.Pool2 = MaxPoolingLayer(pool2, stride2)
        self.Flattener = Flattener()
        self.FullyConnected = FullyConnectedLayer(flattened_height * flattened_width * conv2_channels, n_output_classes)

        self.output = None

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # clear parameter gradients aggregated from the previous pass
        self.params()['Conv1_W'].grad = np.zeros_like(self.params()['Conv1_W'].grad)
        self.params()['Conv1_B'].grad = np.zeros_like(self.params()['Conv1_B'].grad)
        self.params()['Conv2_W'].grad = np.zeros_like(self.params()['Conv2_W'].grad)
        self.params()['Conv2_B'].grad = np.zeros_like(self.params()['Conv2_B'].grad)
        self.params()['FullyConnected_W'].grad = np.zeros_like(self.params()['FullyConnected_W'].grad)
        self.params()['FullyConnected_B'].grad = np.zeros_like(self.params()['FullyConnected_B'].grad)

        # forward pass
        z1 = self.Conv1.forward(X)
        z2 = self.Relu1.forward(z1)
        z3 = self.Pool1.forward(z2)
        z4 = self.Conv2.forward(z3)
        z5 = self.Relu2.forward(z4)
        z6 = self.Pool2.forward(z5)
        z7 = self.Flattener.forward(z6)
        z8 = self.FullyConnected.forward(z7)
        
        # compute loss
        loss, d_out = softmax_with_cross_entropy(z8, y)

        # backward pass
        d_out = self.FullyConnected.backward(d_out)
        d_out = self.Flattener.backward(d_out)
        d_out = self.Pool2.backward(d_out)
        d_out = self.Relu2.backward(d_out)
        d_out = self.Conv2.backward(d_out)
        d_out = self.Pool1.backward(d_out)
        d_out = self.Relu1.backward(d_out)
        d_out = self.Conv1.backward(d_out)

        return loss

    def predict(self, X):
        # forward pass
        z1 = self.Conv1.forward(X)
        z2 = self.Relu1.forward(z1)
        z3 = self.Pool1.forward(z2)
        z4 = self.Conv2.forward(z3)
        z5 = self.Relu2.forward(z4)
        z6 = self.Pool2.forward(z5)
        z7 = self.Flattener.forward(z6)
        z8 = self.FullyConnected.forward(z7)

        pred = np.argmax(z8, axis=1)
        return pred

    def params(self):

        result = {
            'Conv1_W': self.Conv1.W,
            'Conv1_B': self.Conv1.B,
            'Conv2_W': self.Conv2.W,
            'Conv2_B': self.Conv2.B,
            'FullyConnected_W': self.FullyConnected.W,
            'FullyConnected_B': self.FullyConnected.B
        }

        return result
