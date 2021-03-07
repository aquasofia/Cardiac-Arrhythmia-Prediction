from torch import Tensor
from dataset import *
from data_init import *
import json
from sklearn.metrics import multilabel_confusion_matrix, classification_report


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
    y_pred = []
    y_true = []

    # Running the model on the test set
    print('-----------------------------')
    print(' Running model on test set')
    for x, y in loader_testing:
        num_correct = 0
        num_samples = 0
        x = x.squeeze(0)
        # Feed data to the model
        y_hat = model(x)
        # Calculate loss and append to training losses array
        y = torch.LongTensor(y)
        loss_testing = loss_func(y_hat, y)
        losses_testing.append(loss_testing.item())
        print(' loss', loss_testing.item())

        _, predictions = y_hat.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

        # Placeholders for class-wise binary classification
        a = [0, 0, 0, 0, 0]
        b = [0, 0, 0, 0, 0]

        # Construct batch-wise binary map for estimated labels
        for i in predictions:
            a[i.item()] = 1
        y_pred.append(a)
        # Construct batch-wise binary map for ground truth
        for j in y:
            b[j.item()] = 1
        y_true.append(b)

    model.eval()
    # Build a confusion matrix
    confusion_mat = multilabel_confusion_matrix(y_true, y_pred)

    print('\n', 'RESULTS')
    print('-----------------------------')
    print(' Testing loss: ', Tensor(losses_testing).mean().item())
    print(f' Classified in total of {num_correct}/{num_samples} samples')
    print(f' With accuracy of {float(num_correct) / float(num_samples) * 100:.2f}')
    print(' Classification report')
    print(classification_report(y_true, y_pred))



if __name__ == '__main__':
    main()
