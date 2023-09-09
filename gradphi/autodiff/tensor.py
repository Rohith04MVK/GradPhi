import numpy as np


class Tensor:
    def __init__(self, val, args=None, op=None, requires_grad=True, dtype=None):
        if isinstance(val, Tensor):  # If val is a tensor, copy its attributes
            self.data = val.data
            self.grad = val.grad
            self.op = val.op
            self.args = val.args
            self.requires_grad = val.requires_grad
            return

        self.data = np.array(val, dtype=dtype)
        self.grad = None
        self.op = op
        self.args = args
        self.requires_grad = requires_grad

    def tensor(self, t, requires_grad=False):
        return t if isinstance(t, Tensor) else Tensor(t, requires_grad)

    def add(self, t):
        t = self.tensor(t)
        return Tensor(self.data + t.data, [self, t], "add", requires_grad=self.requires_grad or t.requires_grad)

    def sub(self, t):
        t = self.tensor(t)
        return Tensor(self.data - t.data, [self, t], "sub", requires_grad=self.requires_grad or t.requires_grad)

    def mul(self, t):
        t = self.tensor(t)
        return Tensor(self.data * t.data, [self, t], "mul", requires_grad=self.requires_grad or t.requires_grad)

    def div(self, t):
        t = self.tensor(t)
        return Tensor(self.data / t.data, [self, t], "div", requires_grad=self.requires_grad or t.requires_grad)
