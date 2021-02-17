import numpy as np
import wfdb

file_numbers = ['100', '101', '102', '103', '104', '105', '106',
'107', '108', '109', '111', '112', '113', '114', '115', '116', 
'117', '118', '119', '121', '122', '123', '124', '200', '201', 
'202', '203', '205', '207', '208', '209', '210', '212', '213',
'214', '215', '217', '219', '220', '221', '222', '223', '228', 
'230', '231', '232', '233', '234']

filepath = './mit-bih-arrhythmia-database-1.0.0/'

annotations = []
records = []

for num in file_numbers:
    file = filepath + num
    record = wfdb.rdrecord(file, smooth_frames=True)
    annotation = wfdb.rdann(file, 'atr')

    records.append(record)
    annotations.append(annotation)

Fs = 360 # MIT-BIH sampling frequency

# TODO: butterworth
for index, record in enumerate(records):
    signal = record.p_signal
    locations = annotations[index].sample
    # TODO: data to single beats and triplebeat chunks
    for l in locations:
        point = signal[l]
        # do windowing
        # calculate distance to second beat and split from half
        # do dft
        # do bandpass-filtering, 0.1 - 100 Hz / butterworth
        

# TODO: split the data to 5 and 25 min

# TODO: butterworth