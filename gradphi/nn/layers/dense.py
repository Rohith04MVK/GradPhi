import numpy as np

from gradphi.autodiff import Variable


class Dense():
    def __init__(self, in_features, out_features, bias=True):
        # Initialize the super class (Variable)
        super().__init__(None, [], None)

        self.in_features = in_features
        self.out_features = out_features

        stdv = 1. / np.sqrt(in_features)
        self.weight = Variable(
            np.random.uniform(-stdv, stdv, (out_features, in_features)), dtype=np.float32)

        if bias:
            self.bias = Variable(np.zeros((1, out_features)), dtype=np.float32)
        else:
            self.bias = None

    def forward(self, X):
        self.args = [X, self.weight, self.bias]
        self.op = "linear"

        output = np.matmul(X.data, self.weight.data.T)
        if self.bias is not None:
            output += self.bias.data

        # Create a new Dense instance for backpropagation
        return self.backward()

    def backward(self, grad=1):
        if self.bias is not None:
            self.args[2].backward(np.sum(grad, axis=0, keepdims=True))
        self.args[1].backward(
            np.matmul(self.args[0].data.swapaxes(-1, -2), grad).swapaxes(-1, -2))
        self.args[0].backward(np.matmul(grad, self.args[1].data))

    def __call__(self, X):
        return self.forward(X)
