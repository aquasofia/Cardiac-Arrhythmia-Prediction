
from torch import Tensor, tensor
from torch.nn import Module, Conv1d, MaxPool1d, ReLU, Linear, Dropout1d,Sequential


class CNNSystem(Module):

    def __init__(self,input_size:int,
                 dropout: float) \
            -> None:

        super().__init__()

        # First layer : CNN 32 neurons

        self.layer1 = Sequential(Conv1d(input_size, 32, kernel_size=15),
                                 ReLU(),
                                 MaxPool1d(kernel_size=6))

        # 2nd layer : CNN 16 neurons
        self.layer2 = Sequential(Conv1d(32, 16, kernel_size=15),
                                 ReLU(),
                                 MaxPool1d(kernel_size=6))
        
        #3rd layer CNN not kowning neurons
        self.layer3 = Sequential(Conv1d(16, 16, kernel_size=15),
                                 ReLU(),
                                 MaxPool1d(kernel_size=5))
        # need to know input size for 1st mlp layers
        self.mlp1 = Linear(in_features, 10)
        self.mlp2 = Linear(10,5)

        # Dropout
        self.dropout = Dropout1d(dropout)

    def forward(self, x: Tensor) \
            -> Tensor:

        #t = x[0]

        # Neural network layer stacking
        t = self.layer1(x)
        t = self.layer2(t)
        t = self.mlp1(t)



        # t.shape: torch.Size([4, 32, 80, 4]) -> torch.Size([4, 80, 32, 4])
        # Tensor reshaping
        # Permute rotates tensor dimensions preserving the data order
        #a_t = t.permute(0, 2, 1, 3)

        # View maps from one dimensionality to another sequentially
        # Contiguous added for persisting the order during dimensionality change
        # Parameter len is length of the tensor and -1 indicates last member
        #h = a_t.contiguous().view(len(x), -1)
        
        y_hat = self.Dropout1d(self.mlp2(t))

        return y_hat

# EOF
