from dataset import *
from data_init import *
import json
import matplotlib.pyplot as plt
from itertools import cycle


def main():

    # Load the model and training parameters from external file
    configs = json.load(open('configs.json', 'r'))

    # Load training, validation and testing data from directory
    dataset_testing = ECGDataset(data_dir='./testing')

    # Obtain a data loader for testing set
    loader_testing = get_data_loader(
        dataset=dataset_testing,
        batch_size=configs['batch_size'],
        shuffle=True)

    samples_0 = []
    samples_1 = []
    samples_2 = []
    samples_3 = []

    for x, y in loader_testing:
        for i in y:
            if y[i] == 0 and len(samples_0) == 0:
                print('0')
                samples_0 = x.numpy()[0][0][:]
                plt.plot(samples_0[0])
                plt.title('Normal Heartbeat')
                plt.savefig("N.png")
                plt.clf()

            if y[i] == 1 and len(samples_1) == 0:
                print('1')
                samples_1 = x.numpy()[0][0][:]
                plt.plot(samples_1[0])
                plt.title('Supraventricular Ectopic Beat')
                plt.savefig("S.png")
                plt.clf()

            if y[i] == 2 and len(samples_2) == 0:
                print('2')
                samples_2 = x.numpy()[0][0][:]
                plt.plot(samples_2[0])
                plt.title('Ventricular Ectopic Beat')
                plt.savefig("V.png")
                plt.clf()

            if y[i] == 3 and len(samples_3) == 0:
                print('3')
                samples_3 = x.numpy()[0][0][:]
                plt.plot(samples_3[0])
                plt.title('Fusion Beat')
                plt.savefig("F.png")
                plt.clf()

    samples = [samples_0, samples_1, samples_2, samples_3]
    colors = cycle(['blue', 'grey', 'black', 'darkorange'])
    labels = cycle(['N', 'S', 'V', 'F'])
    for j, color, label in zip(range(4), colors, labels):
        plt.plot(samples[j][0], color=color, lw=2, label=f'{label}', )

    plt.legend(loc="lower right")
    plt.savefig("1.png")
    plt.show()


if __name__ == '__main__':
    main()
