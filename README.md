# Cardiac Arrythmia Prediction

A model for classifying different types of cardiac arrythmia. 

## Description

Model consists from following python files:
- cnn (class definition for the cnn)
- dataset (class definition for the dataset)
- training (program for data processing and training functions)
- file_io (program for disk i/o read write operations)
- data_init (program for data initialization)


```bash
pip install torch
pip install tensorflow
```

## Use and development


Model utilizes the PyTorch Librariries:

- Module      for implemetation of the Convolutional Neural Network
- Dataset     to store classification data (features, labels)
- Dataloader  to load data from a custom Dataset object

```python
# retrieve data
dataset =ECGDataset(data_dir='dir/training')

# Obtain a data loader for training set
data = get_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True)

# Dataloader returns features and class for a given index
x, y = data[index]
  
```

## Contribution
For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Copyright
Sofia, Sandra, Sylvia @ tuni 2021
