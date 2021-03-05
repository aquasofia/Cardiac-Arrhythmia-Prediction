from torch.utils.data import Dataset
from torch import tensor, zeros, cat
import numpy as np
import torch


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

        # Pytorch expects input as shape: [N x C x L]
        # N = Number of samples in a batch: Batch one patients all heart beats: 40-60 samples
        # C = Number of channels: 1
        # L = Length of the signal sequence (one heartbeat) : 128
        features = self.key_features[item]
        features = torch.tensor(features, dtype=torch.float)
        # Permute tensor to match pyTorch dimensions
        features = features.permute(0, 2, 1)

        return features, self.key_class[item]

