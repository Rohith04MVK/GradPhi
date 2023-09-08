import numpy as np


class Tensor:
    def __init__(self, data, operation=None, arguments=None, requires_grad=True, data_type=None):
        if isinstance(data, Tensor):
            self.data = data.data
            self.grad = data.grad
            self.operation = data.operation
            self.arguments = data.arguments
            self.requires_grad = data.requires_grad
            return

        self.data = np.array(data, dtype=data_type)
        self.grad = None
        self.operation = operation
        self.arguments = arguments
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"
    

    
