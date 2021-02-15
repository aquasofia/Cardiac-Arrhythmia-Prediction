
from typing import Optional
import torch
from torch.utils.data import DataLoader


def get_data_loader(dataset: torch.utils.data.Dataset,
                    batch_size: Optional[int] = 1,
                    shuffle: Optional[bool] = True) \
        -> DataLoader:
    """Returns Pytorch DataLoader.
    :param dataset: Dataset to iterate over.
    :type dataset: torch.utils.data.Dataset
    :param batch_size: Batch size to use.
    :type batch_size: int
    :param shuffle: Shall we shuffle the examples?
    :type shuffle: bool
    """
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle)

# EOF
