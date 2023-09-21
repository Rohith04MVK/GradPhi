import numpy as np

from gradphi.nn.layers import Dense
# Import the LinearLayer class (assuming it's defined in the same file or module)
# from neunet.autograd import LinearLayer

# Define a simple test function to check the linear layer
def test_linear_layer():
    # Create a LinearLayer instance with 2 input features and 3 output features
    linear_layer = Dense(2, 3)

    # Generate some random input data (batch size of 4)
    X = np.random.rand(4, 2)

    # Forward pass through the linear layer
    output = linear_layer(X)

    # Print the input and output shapes for verification
    print("Input Shape:", X.shape)
    print("Output Shape:", output.shape)

    # Perform a backward pass with a gradient of ones
    output.backward(np.ones_like(output.data))

    # Print the gradients of the weights and bias
    print("Weight Gradients:")
    print(linear_layer.weight.grad)

    if linear_layer.bias is not None:
        print("Bias Gradients:")
        print(linear_layer.bias.grad)

# Run the test
if __name__ == "__main__":
    test_linear_layer()
