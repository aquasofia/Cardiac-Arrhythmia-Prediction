from torch import Tensor
from dataset import *
from data_init import *
import json
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from math import isnan
import time

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

    # Performance evaluation counter
    tic = time.perf_counter()

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
    toc = time.perf_counter()

    print('\n', 'RESULTS')
    print('-----------------------------')
    print(f" Classification took {toc - tic:0.4f} seconds")
    print(' Testing loss: ', Tensor(losses_testing).mean().item())
    print(f' Classified in total of {num_correct}/{num_samples} samples')
    print(f' With accuracy of {float(num_correct) / float(num_samples) * 100:.2f}')
    print(' Classification report')
    print(classification_report(y_true, y_pred))
    print('  Build a confusion matrix')
    print(multilabel_confusion_matrix(y_true, y_pred))

    # Compute ROC curve and ROC area for each class
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Plotting and estimation of FPR, TPR
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for k in range(5):
        fpr[k], tpr[k], _ = roc_curve(y_true[:, k], y_pred[:, k])
        for c in fpr[k]:
            if isnan(c):
                fpr[k] = [0, 0]
        for d in tpr[k]:
            if isnan(d):
                tpr[k] = [1, 1]

        roc_auc[k] = auc(fpr[k], tpr[k])

    colors = cycle(['blue', 'grey', 'black', 'darkorange', 'hotpink'])
    labels = cycle(['N', 'S', 'V', 'F', 'Q'])
    for l, color, label in zip(range(5), colors, labels):
        plt.plot(fpr[l], tpr[l], color=color, lw=2, label=f'ROC for type {label}',)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig("5.png")
    plt.show()


if __name__ == '__main__':
    main()
