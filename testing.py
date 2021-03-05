from torch import Tensor
from dataset import *
from data_init import *
import json


def main():

    # Load the model from file
    model = torch.load('cnn_model')
    model.eval()

    # Load the model and training parameters from external file
    configs = json.load(open('configs.json', 'r'))

    # Load training, validation and testing data from directory
    dataset_testing = ECGDataset(data_dir='./testing')

    # Obtain a data loader for testing set
    loader_testing = get_data_loader(
        dataset=dataset_testing,
        batch_size=configs['batch_size'],
        shuffle=True)

    losses_testing = []

    # For a binary classifier: Logarithmic-Loss and Sigmoid activation
    loss_func = torch.nn.CrossEntropyLoss()

    # Running the model on the test set
    print('-----------------------------')
    print(' Running model on test set')
    for x, y in loader_testing:
        y_hat = model(x).squeeze(1)
        # Calculate loss and append to training losses array
        loss_testing = loss_func(y_hat, y.type_as(y_hat))
        losses_testing.append(loss_testing.item())
        print(' loss', loss_testing.item())

    print('\n', 'RESULTS')
    print(' TESTING LOSS: ', Tensor(losses_testing).mean().item())


if __name__ == '__main__':
    main()
