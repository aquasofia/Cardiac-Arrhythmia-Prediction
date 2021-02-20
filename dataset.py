
from torch.utils.data import Dataset
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
        self.key_class = np.load(data_dir+'/y.npy', allow_pickle=True)

    def __len__(self) \
            -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        """
        # Validate output type, if value = 0 -> return len of hot encodings
        if self.value == 0:
            return len(self.key_hot_encodings)
        # Else return len of key_captions (prevents index overflow)
        else:
            return len(self.key_captions) 
        """

        # Here should implement a length padding for the different length sequences
        return len(self.key_features)

    def __getitem__(self,
                    item: int):
        """Returns an item from the dataset.

        :param item: Index of the item.
        :type item: int
        """

        return self.key_features[item], self.key_class[item]

# EOF

