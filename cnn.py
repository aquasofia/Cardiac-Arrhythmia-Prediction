
from torch import Tensor, tensor
from torch.nn import Module, Conv1d, MaxPool1d, ReLU, Linear, Dropout,Sequential


## needs input dtype= float
class CNNSystem(Module):

    def __init__(self) \
            -> None:

        super().__init__()

        # First layer : CNN xx neurons

        self.layer1 = Sequential(Conv1d(1, 32, kernel_size=15,padding=7),
                                 ReLU(),
                                 MaxPool1d(kernel_size=4))

        # 2nd layer : CNN 32 neurons padding c6
        self.layer2 = Sequential(Conv1d(32, 32, kernel_size=15,padding=7),
                                 ReLU(),
                                 MaxPool1d(kernel_size=6))
        
        #3rd layer CNN 16 neurons
        self.layer3 = Sequential(Conv1d(32, 16, kernel_size=15,padding=7),
                                 ReLU(),
                                 MaxPool1d(kernel_size=5))
        # need to know input size for 1st mlp layers
        self.mlp1 = Linear(16, 10)
        self.mlp2 = Linear(10,5)


    def forward(self, x: Tensor) \
            -> Tensor:

        #t = x[0]

        # Neural network layer stacking
        x = self.layer1(x)

        
        x = self.layer2(x)

        x = x.view(x.size(0), -1)
        t = self.mlp1(x)


        
        y_hat = self.mlp2(x)

        return y_hat

# EOF
