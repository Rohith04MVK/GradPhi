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

    def to_tensor(self, tensor, requires_grad=False):
        return tensor if isinstance(tensor, Tensor) else Tensor(tensor, requires_grad)

    def add(self, tensor):
        tensor = self.to_tensor(tensor)
        return Tensor(self.data + tensor.data, [self, tensor], "add", requires_grad=self.requires_grad or tensor.requires_grad)
