import numpy as np

from gradphi.nn.activations import ReLU
from gradphi.autodiff import Variable

def simple_relu_test():
    relu_layer = ReLU()
    x = Variable(np.array([-2, -1, 0, 1, 2]))
    
    # Test forward pass
    output = relu_layer.forward(x)
    expected_output = np.array([0, 0, 0, 1, 2])
    if np.array_equal(output, expected_output):
        print("\033[92mForward pass: Passed")
    else:
        print("Forward pass: Failed")
    
    # Test backward pass
    relu_layer.forward(x)  # Perform a forward pass first
    backward_output = relu_layer.backward()
    expected_backward = np.array([0, 0, 0, 1, 1])
    if np.array_equal(backward_output, expected_backward):
        print("Backward pass: Passed")
    else:
        print("Backward pass: Failed")

if __name__ == '__main__':
    simple_relu_test()
