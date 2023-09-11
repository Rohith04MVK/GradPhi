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

    def matmul(self, n):
        n = self.to_tensor(n)
        return Variable(np.matmul(self.data, n.data), [self, n], "matmul", requires_grad=self.requires_grad or n.requires_grad)

    def sum(self, *args, **kwargs):
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        return Variable(self.data.sum(*args, **kwargs), [self, axis], "sum", requires_grad=self.requires_grad)

    def mean(self, *args, **kwargs):
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        return Variable(self.data.mean(*args, **kwargs), [self, axis], "mean", requires_grad=self.requires_grad)

    def var(self, *args, **kwargs):
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        return Variable(self.data.var(*args, **kwargs), [self, axis], "var", requires_grad=self.requires_grad)

    def power(self, n):
        n = self.to_tensor(n)
        return Variable(self.data ** n.data, [self, n], "power", requires_grad=self.requires_grad or n.requires_grad)

    def sqrt(self):
        return Variable(np.sqrt(self.data), [self], "sqrt", requires_grad=self.requires_grad)

    def __neg__(self):
        return Variable(-self.data, [self], "neg", requires_grad=self.requires_grad)

    def __pos__(self):
        return Variable(self.data, [self], "pos", requires_grad=self.requires_grad)

    def __abs__(self):
        return self.abs()

    def __add__(self, t):
        return self.add(t)

    def __sub__(self, t):
        return self.sub(t)

    def __mul__(self, t):
        return self.mul(t)

    def __truediv__(self, t):
        return self.div(t)

    def __matmul__(self, t):
        return self.matmul(t)

    def __pow__(self, t):
        return self.power(t)

    def __repr__(self):
        return f"Variable({self.data}, requires_grad={self.requires_grad}, dtype={self.data.dtype})"

    __str__ = __repr__

    def __radd__(self, t):
        t = self.tensor(t)
        return t.add(self)

    def __rsub__(self, t):
        t = self.tensor(t)
        return t.sub(self)

    def __rmul__(self, t):
        t = self.tensor(t)
        return t.mul(self)

    def __rtruediv__(self, t):
        t = self.tensor(t)
        return t.div(self)

    def __rmatmul__(self, t):
        t = self.tensor(t)
        return t.matmul(self)

    def __rpow__(self, t):
        t = self.tensor(t)
        return t.power(self)

    # add unpacking of split tensors
    def __iter__(self):
        return iter(Variable(self.data[i], [self, i], "getitem", requires_grad=self.requires_grad) for i in range(self.data.shape[0]))

    # problem when use grad array indexes: example y[0].grad; non-leaf tensor; in torch it retain_grad
    def __getitem__(self, index):
        return Variable(self.data[index], [self, index], "getitem", requires_grad=self.requires_grad)

    def __array__(self, dtype=None):
        return self.data.astype(dtype, copy=False)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return self.transpose()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data, dtype=self.dtype)
        else:
            grad = np.array(grad, dtype=self.dtype)

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
            self.grad = self.grad + grad

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

        elif self.op == "matmul":

            if self.args[0].data.ndim > 1 and self.args[1].data.ndim > 1:  # [matrix x matrix]
                self.args[0].backward(
                    np.matmul(grad, self.args[1].data.swapaxes(-1, -2)))
                self.args[1].backward(
                    np.matmul(self.args[0].data.swapaxes(-1, -2), grad))

            elif self.args[0].data.ndim == 1 and self.args[1].data.ndim == 1:  # [vector x vector]
                self.args[0].backward(grad * self.args[1].data)
                self.args[1].backward(grad * self.args[0].data)

            elif self.args[0].data.ndim == 1 and self.args[1].data.ndim > 1:  # [vector x matrix]
                self.args[0].backward(grad * self.args[1].data)
                self.args[1].backward(np.outer(grad, self.args[0].data))

            elif self.args[0].data.ndim > 1 and self.args[1].data.ndim == 1:  # [matrix x vector]
                self.args[0].backward(np.outer(grad, self.args[1].data))
                self.args[1].backward(grad * self.args[0].data)

        elif self.op == "sum":
            axis = self.args[1]
            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.args[0].backward(np.ones_like(self.args[0].data) * grad)

        elif self.op == "mean":
            axis = self.args[1]

            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = np.expand_dims(grad, axis)

            _axis = list(axis) if isinstance(axis, tuple) else axis
            self.args[0].backward(np.ones_like(
                self.args[0].data) * grad / np.prod(np.array(self.args[0].data.shape)[_axis]))

        elif self.op == "var":
            axis = self.args[1]

            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = np.expand_dims(grad, axis)
            _axis = list(axis) if isinstance(axis, tuple) else axis
            self.args[0].backward(np.ones_like(self.args[0].data) * grad * 2 * (self.args[0].data - self.args[0].data.mean(
                axis=axis, keepdims=True)) / np.prod(np.array(self.args[0].data.shape)[_axis]))

        elif self.op == "power":
            self.args[0].backward(
                grad * self.args[1].data * self.args[0].data ** (self.args[1].data - 1))
            self.args[1].backward(grad * self.args[0].data **
                                  self.args[1].data * np.log(self.args[0].data))

        elif self.op == "sqrt":
            self.args[0].backward(grad * 1 / (2 * np.sqrt(self.args[0].data)))
