import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input: int, n_output: int, hidden_layer_size: int, reg: float):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X: np.array, y: np.array) -> float:
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # clear parameter gradients aggregated from the previous pass
        # Hint: using self.params() might be useful!
        self.params()['W1'].grad = np.zeros_like(self.params()['W1'].grad)
        self.params()['B1'].grad = np.zeros_like(self.params()['B1'].grad)
        self.params()['W2'].grad = np.zeros_like(self.params()['W2'].grad)
        self.params()['B2'].grad = np.zeros_like(self.params()['B2'].grad)

        # forward pass
        z = self.fc1.forward(X)
        z1 = self.relu.forward(z)
        z2 = self.fc2.forward(z1)

        # compute loss
        loss, d_out = softmax_with_cross_entropy(z2, y) # head of the model
        
        # backward pass
        d_out = self.fc2.backward(d_out)
        d_out = self.relu.backward(d_out)
        d_out = self.fc1.backward(d_out)
        
        # l2 regularization on all params
        # Hint: self.params() is useful again!
        loss_reg1, grad_reg1 = l2_regularization(self.params()['W1'].value, self.reg)
        loss_reg2, grad_reg2 = l2_regularization(self.params()['W2'].value, self.reg)
        loss += loss_reg1 + loss_reg2

        self.params()['W1'].grad += grad_reg1
        self.params()['W2'].grad += grad_reg2

        return loss

    def predict(self, X: np.array) -> np.array:
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

        z = self.fc1.forward(X)
        z1 = self.relu.forward(z)
        z2 = self.fc2.forward(z1)

        pred = np.argmax(z2, axis=1)

        return pred

    def params(self):
        result = {
            'W1': self.fc1.W,
            'B1': self.fc1.B,
            'W2': self.fc2.W,
            'B2': self.fc2.B,
        }

        return result
