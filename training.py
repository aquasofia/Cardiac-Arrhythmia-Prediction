from cnn import CNNSystem
from torch import Tensor
from dataset import *
from data_init import *
import json
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import time


def main():

    # Load training data from the directory
    dataset_training = ECGDataset(data_dir='./training')
    # dataset_training_chunks = ECGDataset(data_dir='./training/chunks')
    # dataset_validation = ECGDataset(0, 0, data_dir='validation')

    # Load the model and training parameters from external file
    configs = json.load(open('configs.json', 'r'))

    # Obtain a data loader for training set
    loader_training = get_data_loader(
        dataset=dataset_training,
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
    loss_func = torch.nn.CrossEntropyLoss()
    # Initialize arrays for losses
    losses_training = []

    # Measure performance for 100 epochs
    tic = time.perf_counter()
    num_correct = 0
    num_samples = 0

    # Set training loop to max epoch
    for epoch in range(configs['max_epochs']):
        # 1. Training the neural network with training set
        print('-----------------------------')
        print(' Running model in training set')

        for x, y in loader_training:
            # Reset gradients from the previous round
            optimizer.zero_grad()
            x = x.squeeze(0)
            # Feed data to the model
            y_hat = cnn(x)
            # Calculate loss and append to training losses array
            y = torch.LongTensor(y)
            loss_training = loss_func(y_hat, y)
            losses_training.append(loss_training.item())
            print(' loss', loss_training.item())
            # Initiate backpropagation on the basis of the loss
            loss_training.backward()
            # Optimize network weights
            optimizer.step()
            _, predictions = y_hat.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    cnn.eval()
    toc = time.perf_counter()

    print('\n', 'RESULTS')
    print('-----------------------------')
    print(f" Model training time {toc - tic:0.4f} seconds")
    print(' Training loss: ', Tensor(losses_training).mean().item())
    print(f' Classified in total of {num_correct}/{num_samples} samples')
    print(f' With accuracy of {float(num_correct) / float(num_samples) * 100:.2f}')
    print('-----------------------------')
    print('\n', 'EPOCH ', epoch, '| LOSS MEAN ', Tensor(losses_training).mean().item())

    torch.save(cnn, 'cnn_model')


if __name__ == '__main__':
    main()
