
from torch import Tensor
from torch.nn import Module, Conv2d, MaxPool2d, ReLU, Linear, Dropout2d


class CNNSystem(Module):

    def __init__(self,
                 layer_1_input_dim: int,
                 layer_1_output_dim: int,
                 layer_2_input_dim: int,
                 layer_2_output_dim: int,
                 pooling_1_kernel: int,
                 pooling_1_stride: int,
                 pooling_2_kernel: int,
                 pooling_2_stride: int,
                 kernel_1: int,
                 kernel_2: int,
                 stride_1: int,
                 stride_2: int,
                 padding_1: int,
                 padding_2: int,
                 input_features: int,
                 output_features: int,
                 dropout: float) \
            -> None:

        super().__init__()

        # First layer : CNN
        self.layer_1 = Conv2d(in_channels=layer_1_input_dim,
                              out_channels=layer_1_output_dim,
                              kernel_size=kernel_1,
                              stride=stride_1,
                              padding=padding_1)

        # Third layer : CNN
        self.layer_2 = Conv2d(in_channels=layer_2_input_dim,
                              out_channels=layer_2_output_dim,
                              kernel_size=kernel_2,
                              stride=stride_2,
                              padding=padding_2)

        # Second layer : Pooling
        self.pooling_1 = MaxPool2d(kernel_size=pooling_1_kernel,
                                   stride=pooling_1_stride)

        # Fourth layer : Pooling
        self.pooling_2 = MaxPool2d(kernel_size=pooling_2_kernel,
                                   stride=pooling_2_stride)

        # Classifier
        self.classifier = Linear(in_features=input_features,
                                 out_features=output_features)
        # Activation function
        self.relu_1 = ReLU()
        self.relu_2 = ReLU()

        # Dropout
        self.dropout = Dropout2d(dropout)

    def forward(self,
                x: Tensor) \
            -> Tensor:

        # Dataset dimensionality correction, if less than 4
        if x.ndimension() != 4:
            t = x.unsqueeze(1)

        # Neural network layer stacking
        t = self.pooling_1(self.relu_1(self.layer_1(t)))
        t = self.pooling_2(self.relu_2(self.layer_2(t)))

        # Adding dropout
        t = self.dropout(t)

        # t.shape: torch.Size([4, 32, 80, 4]) -> torch.Size([4, 80, 32, 4])
        # Tensor reshaping
        # Permute rotates tensor dimensions preserving the data order
        a_t = t.permute(0, 2, 1, 3)

        # View maps from one dimensionality to another sequentially
        # Contiguous added for persisting the order during dimensionality change
        # Parameter len is length of the tensor and -1 indicates last member
        h = a_t.contiguous().view(len(x), -1)
        y_hat = self.classifier(h)

        return y_hat

# EOF
