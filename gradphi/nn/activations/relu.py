import numpy as np


class ReLU():
    def __init__(self):
        self.input_data = None

    def forward(self, x):
        self.input_data = x
        f_x = np.maximum(0, x)
        return f_x

    def backward(self, grad=1):
        if self.input_data is None:
            raise ValueError("Forward must be called before backward.")
        return grad * (self.input_data > 0)

    def __call__(self, x):
        return self.forward(x)
