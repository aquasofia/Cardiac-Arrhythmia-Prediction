from os import path
import numpy as np
import torch
from torch import Tensor
from dataset import *
from data_init import *
from cnn import CNNSystem
import preprocessing
import json


def main():

    # Load training, validation and testing data from directory
    dataset_training = ECGDataset(data_dir='./training')
    dataset_testing = ECGDataset(data_dir='./testing')
    #dataset_training_chunks = ECGDataset(data_dir='./training/chunks')
    # dataset_validation = ECGDataset(0, 0, data_dir='validation')

    # Load the model and training parameters from external file
    configs = json.load(open('configs.json', 'r'))

    # Obtain a data loader for training set
    loader_training = get_data_loader(
        dataset=dataset_training,
        batch_size=configs['batch_size'],
        shuffle=True)

    # Obtain a data loader for testing set
    loader_testing = get_data_loader(
        dataset=dataset_testing,
        batch_size=configs['batch_size'],
        shuffle=True)

    # Create an instance of the CNN model
    cnn = CNNSystem(layer_1_input_dim=configs['model']['layer_1_input_dim'],
                    layer_1_output_dim=configs['model']['layer_1_output_dim'],
                    layer_2_input_dim=configs['model']['layer_2_input_dim'],
                    layer_2_output_dim=configs['model']['layer_2_output_dim'],
                    pooling_1_kernel=configs['model']['pooling_1_kernel'],
                    pooling_2_kernel=configs['model']['pooling_2_kernel'],
                    pooling_1_stride=configs['model']['pooling_1_stride'],
                    pooling_2_stride=configs['model']['pooling_2_stride'],
                    input_features=configs['model']['input_features'],
                    output_features=configs['model']['classifier_output'],
                    kernel_1=configs['model']['kernel_1'],
                    kernel_2=configs['model']['kernel_2'],
                    stride_1=configs['model']['stride_1'],
                    stride_2=configs['model']['stride_2'],
                    padding_1=configs['model']['padding_1'],
                    padding_2=configs['model']['padding_2'],
                    dropout=configs['model']['dropout'])

    # For a binary classifier ADAM optimizer
    optimizer = torch.optim.Adam(cnn.parameters())
    # For a binary classifier: Logarithmic-Loss and Sigmoid activation
    loss_func = torch.nn.BCEWithLogitsLoss()
    # Initialize arrays for losses
    losses_training = []
    losses_validation = []
    losses_testing = []
    # Initialize minimum_loss to infinity (arbitrary high number)
    minimum_loss = np.inf
    # Initialize parameters for early stop
    not_improved = 0
    epoch_stop = 5

    # Set training loop to max epoch
    for epoch in range(configs['max_epochs']):
        # 1. Training the neural network with training set
        print('-----------------------------')
        print(' Running model in training set')

        for x, y in loader_training:
            # Reset gradients from the previous round
            optimizer.zero_grad()
            x = x.squeeze(0)
            # Remove the batch dimension on 0 as batch = 1
            y_hat = cnn(x)
            # Calculate loss and append to training losses array
            loss_training = loss_func(y_hat, y.type_as(y_hat))
            losses_training.append(loss_training.item())
            print(' loss', loss_training.item())
            # Initiate backpropagation on the basis of the loss
            loss_training.backward()
            # Optimize network weights
            optimizer.step()
        cnn.eval()
        print('-----------------------------')
        print('\n', 'EPOCH ', epoch, '| LOSS MEAN ', Tensor(losses_training).mean().item())

        # Running the model on the test set
        print('-----------------------------')
        print(' Running model on test set')
        for x, y in loader_testing:
            # Reset gradients from the previous round
            optimizer.zero_grad()
            y_hat = cnn_final(x).squeeze(1)
            # Calculate loss and append to training losses array
            loss_testing = loss_func(y_hat, y.type_as(y_hat))
            losses_testing.append(loss_testing.item())
            print(' loss', loss_testing.item())

        print('-----------------------------')
        print('\n', 'EPOCH ', epoch, '| LOSS MEAN ', Tensor(losses_testing).mean().item())

    print('\n', 'RESULTS')
    print(' TRAINING LOSS: ', Tensor(losses_training).mean().item(), ' | ',
          ' VALIDATION LOSS: ',Tensor(losses_validation).mean().item(), ' | ',
          ' TESTING LOSS: ', Tensor(losses_testing).mean().item())


if __name__ == '__main__':
    main()
