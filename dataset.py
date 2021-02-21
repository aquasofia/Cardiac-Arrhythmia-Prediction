from torch.utils.data import Dataset
from torch import tensor, zeros, cat
import numpy as np


class ECGDataset(Dataset):

    def __init__(self,
                 data_dir: str):
        """An example of an object of class torch.utils.data.Dataset

        :type key_hot_encodings: str
        :param key_words: Key to use for getting the class, defaults\
                          to `class`.
        :type key_words: str
        """
        super().__init__()
        self.key_features = np.load(data_dir + '/X.npy', allow_pickle=True)
        self.key_class = np.load(data_dir + '/y.npy', allow_pickle=True)

    def __len__(self) \
            -> int:
        """
        Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """

        return len(self.key_features)

    def __getitem__(self,
                    item: int):

        """Returns an item from the dataset.
        :param item: Index of the item.
        :type item: int
        """

        return self.key_features[item], self.key_class[item],

"""
def collate_fn(data):

    data: is a list of tuples with (example, label, length)
    where 'example' is a tensor of arbitrary shape
    and label/length are scalars

 _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    return features.float(), labels.long(), lengths.long()
    
"""