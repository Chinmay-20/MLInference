import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron, feed-forward neural network
    Input dimension: 10
    Hidden layer: each layer has size 20
    Output: single value as output.
    """
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 1):
        super(SimpleMLP, self).__init__()
        # Build layer pipeline:
        '''
        Input(10) -> Linear(10 -> 20) -> ReLU -> Linear(20 -> 20) ->
        -> ReLU -> Linear (20 -> 1) Output
        '''
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network. Define: Input -> network -> Output
        Automatically called with model(x)
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output prediction
        """
        return self.network(x)

if __name__ == "__main__":
    model = SimpleMLP()
    dummy_input = torch.randn(1, 10)    # Random tensor with size (1,10) 
                                        # -> one rows, 10 columns.
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")