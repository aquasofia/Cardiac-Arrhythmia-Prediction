
from torch import Tensor, tensor
from torch.nn import Module, Conv1d, MaxPool1d, ReLU, Linear, Dropout, Sequential


## needs input dtype= float
class CNNSystem(Module):

    def __init__(self,
                 layer_1_input_dim,
                 layer_1_output_dim,
                 layer_2_input_dim,
                 layer_2_output_dim,
                 pooling_1_kernel,
                 pooling_2_kernel,
                 pooling_1_stride,
                 pooling_2_stride,
                 input_features,
                 output_features,
                 kernel_1,
                 kernel_2,
                 stride_1,
                 stride_2,
                 padding_1,
                 padding_2,
                 dropout
                 ) \
            -> None:

        super().__init__()

        # First layer : CNN xx neurons

        self.layer1 = Sequential(Conv1d(in_channels=1, out_channels=32, kernel_size=15, padding=7),
                                 ReLU(),
                                 MaxPool1d(kernel_size=4))

        # 2nd layer : CNN 32 neurons padding c6
        self.layer2 = Sequential(Conv1d(32, 32, kernel_size=15, padding=7),
                                 ReLU(),
                                 MaxPool1d(kernel_size=6))
        
        #3rd layer CNN 16 neurons
        self.layer3 = Sequential(Conv1d(32, 16, kernel_size=15, padding=7),
                                 ReLU(),
                                 MaxPool1d(kernel_size=5))

        # need to know input size for 1st mlp layers
        self.mlp1 = Linear(16, 10)
        self.mlp2 = Linear(10,5)

    def forward(self, x: Tensor) \
            -> Tensor:

        #t = x[0]
        x = x if x.ndimension() == 4 else x.unsqueeze(1)

        # Neural network layer stacking
        x = self.layer1(x)

        x = self.layer2(x)

        x = x.view(x.size(0), -1)
        t = self.mlp1(x)
        
        y_hat = self.mlp2(x)

        return y_hat

# EOF
