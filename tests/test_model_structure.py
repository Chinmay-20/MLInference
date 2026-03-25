import torch
import pytest

# it works thanks to pytest.ini
from ml_src.model import SimpleMLP

def test_model_input_output_shape():
    """
    Verifies that the model accepts the correct input shape 
    and produces the expected output shape.
    """
    # initialize model
    model = SimpleMLP(input_size=10, output_size=1)
    
    # create a dummy batch of size 32
    batch_size = 32
    input_features = 10
    dummy_input = torch.randn(batch_size, input_features)
    
    # forward pass
    output = model(dummy_input)
    
    # Check 1: Output should not be None
    assert output is not None
    
    # Check 2: Output shape should be (Batch_Size, Output_Size)
    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape

def test_model_parameters_update():
    """
    Checks if model parameters have gradients, that is, are trainable.
    """
    model = SimpleMLP()
    dummy_input = torch.randn(1, 10)
    output = model(dummy_input)
    
    # Simulate a loss: we just need a scalar value to perform the backpropagation
    loss = output.mean()

    # Apply chain rule for each model parameter.
    loss.backward()
    
    # Check if gradients are populated, that is, if each parameters participates 
    # in the graph.
    for param in model.parameters():
        assert param.grad is not None