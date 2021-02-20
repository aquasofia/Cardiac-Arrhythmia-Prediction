from os import path
import numpy as np
import torch
from torch import Tensor
from dataset import ECGDataset
import data_init
from cnn import CNNSystem


def main():
    # Load training, validation and testing data from directory
    dataset_training = ECGDataset(data_dir='./training')
    dataset_testing = ECGDataset(data_dir='./testing')
    dataset_training_chunks = ECGDataset(data_dir='./training/chunks')
    #dataset_validation = ECGDataset(0, 0, data_dir='validation')

    # Define training parameters
    training_parameters = {'batch_size': 4,
                           'shuffle': True,
                           'max_epochs': 100}

    # Define parameters for the CNN model
    cnn_parameters = {'layer_1_input_dim': 1,
                      'layer_1_output_dim': 16,
                      'kernel_1': 3,
                      'stride_1': 2,
                      'padding_1': 2,

                      'pooling_1_kernel': 4,
                      'pooling_1_stride': 1,

                      'layer_2_input_dim': 16,
                      'layer_2_output_dim': 32,
                      'kernel_2': 3,
                      'stride_2': 2,
                      'padding_2': 2,

                      'pooling_2_kernel': 4,
                      'pooling_2_stride': 2,

                      'classifier_output': 1,
                      'input_features': 10240,
                      'dropout': 0.1}

    # Obtain a data loader for training set
    loader_training = get_data_loader(
        dataset=dataset_training,
        batch_size=training_parameters['batch_size'],
        shuffle=True)

    # Obtain a data loader for validation set
    loader_validation = get_data_loader(
        dataset=dataset_validation,
        batch_size=training_parameters['batch_size'],
        shuffle=True)

    # Obtain a data loader for testing set
    loader_testing = get_data_loader(
        dataset=dataset_testing,
        batch_size=training_parameters['batch_size'],
        shuffle=True)

    # Create an instance of the CNN model
    cnn = CNNSystem(layer_1_input_dim=cnn_parameters['layer_1_input_dim'],
                      layer_1_output_dim=cnn_parameters['layer_1_output_dim'],
                      layer_2_input_dim=cnn_parameters['layer_2_input_dim'],
                      layer_2_output_dim=cnn_parameters['layer_2_output_dim'],
                      pooling_1_kernel=cnn_parameters['pooling_1_kernel'],
                      pooling_2_kernel=cnn_parameters['pooling_2_kernel'],
                      pooling_1_stride=cnn_parameters['pooling_1_stride'],
                      pooling_2_stride=cnn_parameters['pooling_2_stride'],
                      input_features=cnn_parameters['input_features'],
                      output_features=cnn_parameters['classifier_output'],
                      kernel_1=cnn_parameters['kernel_1'],
                      kernel_2=cnn_parameters['kernel_2'],
                      stride_1=cnn_parameters['stride_1'],
                      stride_2=cnn_parameters['stride_2'],
                      padding_1=cnn_parameters['padding_1'],
                      padding_2=cnn_parameters['padding_2'],
                      dropout=cnn_parameters['dropout'])

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
    for epoch in range(training_parameters['max_epochs']):
        # 1. Training the neural network with training set
        print('-----------------------------')
        print(' Running model in training set')
        for x, y in loader_training:
            # Reset gradients from the previous round
            optimizer.zero_grad()
            y_hat = cnn(x).squeeze(1)
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

        # 2. Validating performance with a validation set
        print('-----------------------------')
        print(' Running model in validation set')
        for x, y in loader_validation:
            # Reset gradients from the previous round
            optimizer.zero_grad()
            y_hat = cnn(x).squeeze(1)
            # Calculate loss and append to training losses array
            loss_validation = loss_func(y_hat, y.type_as(y_hat))
            losses_validation.append(loss_validation.item())
            print(' loss', loss_validation.item())

        print('-----------------------------')
        print('\n', ' EPOCH ', epoch, '| LOSS MEAN ', Tensor(losses_validation).mean().item())

        # 3. Investigating the model performance and performing early stopping
        if loss_validation < minimum_loss:
            minimum_loss = loss_validation
            not_improved = 0
            torch.save(cnn, 'model')
        else:
            not_improved += 1

        if epoch > 5 and not_improved == epoch_stop:
            print(' Early stopping')
            break

        if path.exists('model'):
            cnn_final = torch.load('model')

        # 4. Running the model on the test set
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
    print(' TRAINING LOSS: ', Tensor(losses_training).mean().item(), ' | '
          ' VALIDATION LOSS: ', Tensor(losses_validation).mean().item(), ' | '
          ' TESTING LOSS: ', Tensor(losses_testing).mean().item())

if __name__ == '__main__':
    main()
