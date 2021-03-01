from pathlib import Path
import numpy as np
import wfdb
from scipy import signal
import os

file_numbers = ['100', '101', '102', '103', '104', '105', '106',
'107', '108', '109', '111', '112', '113', '114', '115', '116', 
'117', '118', '119', '121', '122', '123', '124', '200', '201', 
'202', '203', '205', '207', '208', '209', '210', '212', '213',
'214', '215', '217', '219', '220', '221', '222', '223', '228', 
'230', '231', '232', '233', '234']

#individual_file = ['100']

filepath = './mit-bih-arrhythmia-database-1.0.0/'


Fs = 360 # MIT-BIH sampling frequency
beat_length = 128

# Bandpass butterworth filter, passband 0,5 Hz - 40 Hz. 
# Passband width is used in GAN-study
sos  = signal.butter(20, [0.5, 40], 'bandpass', fs=Fs, output='sos')

def read_data(sampfrom, sampto):
    records = []
    annotations = []
    for num in file_numbers:
        file = filepath + num
        record = wfdb.rdrecord(file, sampfrom=sampfrom, sampto=sampto, channels=[0]) # Lead 1
        annotation = wfdb.rdann(file, 'atr', sampfrom=sampfrom, sampto=sampto)
        records.append(record)
        annotations.append(annotation)
    
    return records, annotations

def process_data(records, annotations):
    splitsignals = []
    for i, record in enumerate(records):
        p_signal = record.p_signal
        locations = annotations[i].sample
        splitsignal = []
        last_cutoff = 0
        for j in range(len(locations) - 1):

            if (j == 0):
                d_prev = 0
            else:
                d_prev = round((locations[j] - locations[j - 1]))

            d_next = round(locations[j + 1] - locations[j])
            interval_length = d_prev + (d_next - d_prev)
            interval_third = round(interval_length / 3)

            first_cutoff = int(locations[j] - interval_third)
            second_cutoff = int(locations[j] + 2 * interval_third)

            if (second_cutoff >= locations[-1]):
                part = p_signal[first_cutoff:]
            else:
                part = p_signal[first_cutoff:second_cutoff]

            filtered = signal.sosfilt(sos, part)

            s = filtered.ravel()
            #s = filtered/max(s)
            splitsignal.append(s)
        splitsignals.append(splitsignal)
    
    return splitsignals

def reshape(data):
    for sample in data:
        for i, beat in enumerate(sample):
            if (len(beat) > beat_length or len(beat < beat_length)):
               resampled = signal.resample(beat, beat_length)
               sample[i] = resampled
    return data

def create_three_beat_chunks(data):
    chunks = []
    for patient_sample in data:
        patient_chunks = []
        for i in range(len(patient_sample)-1):
            if (i == 0) :
                chunk = np.concatenate((patient_sample[i], patient_sample[i+1], patient_sample[i+2]))
            if (i == len(patient_sample)-1):
                chunk = np.concatenate((patient_sample[i-2], patient_sample[i-1], patient_sample[i]))
            else:
                chunk = np.concatenate((patient_sample[i-1], patient_sample[i], patient_sample[i+1]))
            patient_chunks.append(chunk)
        chunks.append(patient_chunks)
    
    return chunks

def group_to_five_classes(annot):
    # N, S, V, F, Q
    unknown = [0, 12, 14, 15, 16, 17, 18, 19, 20,
     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     32, 33, 35, 36, 37, 38, 39, 40, 41]

    for patient_annot in annot:
        for i, num in enumerate(patient_annot.num):
            if (num == 2 or num == 3 or num == 11 or num == 34): # N = 1
                patient_annot.num[i] = 1
            elif (num == 4 or num == 8 or num == 7): # S = 9
                patient_annot.num[i] = 9
            elif (num == 31 or num == 10): # V = 5
                patient_annot.num[i] = 5
            elif (num == 6): # F = 6
                patient_annot.num[i] = 6
            else: # Q = 13
                patient_annot.num[i] = 13
        
    return annot

def save_data(file, arr):
    np.save(file, arr)

def save_data(file, arr):
    path = '/'.join(file.split('/')[:-1])
    if not Path(path).exists():
        os.mkdir(path)
    np.save(file, arr)


def save_labels(file, arr):
    labels = []
    for patient_annot in arr:
        labels.append(patient_annot.num)
    labels = np.save(file, labels)


def plot_data(s):
    wfdb.plot_items(s)


def main():
    training_data, training_annotations = read_data(0, 10799)
    testing_data, testing_annotations = read_data(10800, 647999)

    split_training_data = process_data(training_data, training_annotations)
    split_testing_data = process_data(testing_data, testing_annotations)

    
    training_data = reshape(split_training_data)
    testing_data = reshape(split_testing_data)
    #plot_data(training_data[0][4])

    training_chunks = create_three_beat_chunks(training_data)

    training_annotations = group_to_five_classes(training_annotations)
    testing_annotations = group_to_five_classes(testing_annotations)

    save_data('./training/X', training_data)
    save_data('./testing/X', testing_data)

    save_data('./training/chunks/X', training_chunks)
    save_labels('./training/chunks/y', training_annotations)

    save_labels('./training/y', training_annotations)
    save_labels('./testing/y', testing_annotations)
    


if __name__ == "__main__":
    main()
