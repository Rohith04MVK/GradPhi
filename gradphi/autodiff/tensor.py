import numpy as np


class Variable:
    def __init__(self, val, args=None, op=None, requires_grad=True, dtype=None):
        if isinstance(val, Variable):
            self.data = val.data
            self.grad = val.grad
            self.op = val.op
            self.args = val.args
            self.requires_grad = val.requires_grad
            self.dtype = val.dtype
            return

        self.data = np.array(val, dtype=dtype)
        self.grad = None
        self.op = op
        self.args = args
        self.requires_grad = requires_grad
        self.dtype = dtype

    def to_tensor(self, t, requires_grad=False):
        return t if isinstance(t, Variable) else Variable(t, requires_grad)

    def add(self, t):
        t = self.to_tensor(t)
        return Variable(self.data + t.data, [self, t], "add", requires_grad=self.requires_grad or t.requires_grad)

    def sub(self, t):
        t = self.to_tensor(t)
        return Variable(self.data - t.data, [self, t], "sub", requires_grad=self.requires_grad or t.requires_grad)

    def mul(self, t):
        t = self.to_tensor(t)
        return Variable(self.data * t.data, [self, t], "mul", requires_grad=self.requires_grad or t.requires_grad)

    def div(self, t):
        t = self.to_tensor(t)
        return Variable(self.data / t.data, [self, t], "div", requires_grad=self.requires_grad or t.requires_grad)

    def __add__(self, t):
        return self.add(t)

    def __sub__(self, t):
        return self.sub(t)

    def __mul__(self, t):
        return self.mul(t)

    def __truediv__(self, t):
        return self.div(t)

    def backward(self, grad=None):  # grad=np.array(1) # TODO: ASSERT GRAD SHAPE == DATA SHAPE
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data, dtype=self.dtype)
        else:
            grad = np.array(grad, dtype=self.dtype)

        # reverse broadcast; TODO : MAYBE MOVE IT TO ANOTHER PLACE
        if grad.size != self.data.size or grad.ndim != self.data.ndim or grad.shape != self.data.shape:
            if self.data.size == 1:
                grad = grad.sum()
            elif self.data.ndim == grad.ndim:
                grad = grad.sum(axis=tuple(
                    np.where(np.array(self.data.shape) != np.array(grad.shape))[0]), keepdims=True)
            else:
                data_shape = (1,) * (grad.ndim -
                                     self.data.ndim) + self.data.shape
                axis = tuple(np.where(np.array(data_shape)
                             != np.array(grad.shape))[0])
                grad = grad.sum(axis=axis).reshape(self.data.shape)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad  # += BUG FIX

        if self.op == "add":
            self.args[0].backward(grad)
            self.args[1].backward(grad)

        elif self.op == "sub":
            self.args[0].backward(grad)
            self.args[1].backward(-grad)

        elif self.op == "mul":
            self.args[0].backward(grad * self.args[1].data)
            self.args[1].backward(self.args[0].data * grad)

        elif self.op == "div":
            self.args[0].backward(grad / self.args[1].data)
            self.args[1].backward(-grad * self.args[0].data /
                                  self.args[1].data ** 2)
