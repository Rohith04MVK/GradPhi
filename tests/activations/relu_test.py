import numpy as np
from gradphi.nn.activations import ReLU
from gradphi.autodiff import Variable


def test_relu_forward():
    x = Variable(np.array([-2, -1, 0, 1, 2]))
    relu_layer = ReLU(x)
    output = relu_layer.forward()
    expected_output = np.array([0, 0, 0, 1, 2])
    assert np.array_equal(output, expected_output)


def test_relu_backward():
    x = Variable(np.array([-2, -1, 0, 1, 2]))
    relu_layer = ReLU(x)
    _ = relu_layer.forward()  # Forward pass
    backward_output = relu_layer.backward()
    expected_backward = np.array([0, 0, 1, 1, 1])
    assert np.array_equal(backward_output, expected_backward)
