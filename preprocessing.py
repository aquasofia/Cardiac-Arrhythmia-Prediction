from pathlib import Path
import numpy as np
import wfdb
from scipy import signal
import os
import random

all_files = ['100', '101', '103', '105', '106',
'108', '109', '111', '112', '113', '114', '115', '116', 
'117', '118', '119', '121', '122', '123', '124', '200', '201', 
'202', '203', '205', '207', '208', '209', '210', '212', '213',
'214', '215', '219', '220', '221', '222', '223', '228', 
'230', '231', '232', '233', '234']

common_training_data = ['100', '101', '103', '105', '106',
'108', '109', '111', '112', '113', '114', '115', '116', 
'117', '118', '119', '121', '122', '123', '124']

#paced = ['102', '104', '107', '217'] #these are excluded from the data

#individual_file = ['100']

filepath = './mit-bih-arrhythmia-database-1.0.0/'

Fs = 360 # MIT-BIH sampling frequency
beat_length = 128

# Bandpass butterworth filter, passband 0,5 Hz - 40 Hz. 
# Passband width is used in GAN-study
sos  = signal.butter(20, [0.5, 40], 'bandpass', fs=Fs, output='sos')

def read_data(sampfrom, sampto, filegroup):
    records = []
    annotations = []
    for num in filegroup:
        file = filepath + num
        record = wfdb.rdrecord(file, sampfrom=sampfrom, sampto=sampto, channels=[0]) # Lead 1
        annotation = wfdb.rdann(file, 'atr', sampfrom=sampfrom, sampto=sampto)
        records.append(record)
        annotations.append(annotation)

    return records, annotations


def process_data(records, annotations, testing=False):
    splitsignals = []
    for i, record in enumerate(records):
        p_signal = record.p_signal
        locations = annotations[i].sample
        splitsignal = []
        last_cutoff = 0
        for j in range(len(locations) - 1):

            R_peak = locations[j] 
            next_R_peak = locations[j + 1]
            if (testing):
                R_peak = R_peak - 10800
                next_R_peak = next_R_peak - 10800

            if (j == 0):
                # distance to the start of the ECG
                #print(record.record_name + " " + str(len(p_signal)) + " " + str(R_peak))
                distance_prev_beat = int(round(R_peak))
            else:
                # distance to the previous R-peak
                if (testing):
                    last_R_peak = locations[j - 1] - 10800
                else:
                    last_R_peak = locations[j - 1]
                distance_prev_beat = int(round(R_peak - last_R_peak))

            # distance to next R-peak
            distance_next_beat = int(round(next_R_peak - R_peak))

            # lenght of the interval from previous R-peak to next R-peak
            interval_length = distance_prev_beat + (distance_next_beat - distance_prev_beat)
            interval_third = round(interval_length / 3)

            if (j == 0 or interval_third > R_peak):
                first_cutoff = 0
            else:
                first_cutoff = int(R_peak - interval_third)
            
            second_cutoff = int(R_peak + 2 * interval_third)

            if (second_cutoff >= locations[-1] or second_cutoff > len(p_signal)):
                part = p_signal[first_cutoff:]
            else:
                part = p_signal[first_cutoff:second_cutoff]
                #if (len(part) == 0):
                #    print(record.record_name + " " + str(locations[j]) + " second " + str(p_signal[second_cutoff]) +  " " + str(first_cutoff)+ ":" + str(second_cutoff))

            filtered = signal.sosfilt(sos, part)

            s = filtered.ravel()
            s = filtered/max(np.abs(s))
            splitsignal.append(s)
        #print(len(splitsignal))
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


def group_to_five_classes(data, annot):
    # N, S, V, F, Q
    unknown = [0, 12, 14, 15, 16, 17, 18, 19, 20,
     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     32, 33, 35, 36, 37, 38, 39, 40, 41]

    numerical_annotations = []
    removed_samples = []
    for j, patient_annot in enumerate(annot):
        annotations = []
        for i, sym in enumerate(patient_annot.symbol):
            if (sym == 'N' or sym == 'L' or sym == 'R' or sym == 'j' or sym == 'e'): # N = 1
                annotations.append(0)
            elif (sym == 'S' or sym == 'a' or sym == 'J' or sym == 'A'): # S = 9
                annotations.append(1)
            elif (sym == 'V' or sym == '!' or sym == 'E'): # V = 5
                annotations.append(2)
            elif (sym == 'F'): # F = 6
                annotations.append(3)
            elif (sym == 'Q'): # Q = 13
                annotations.append(4)
            else:
                removed_samples.append([j,i])
        del annotations[-1]
        numerical_annotations.append(annotations)
    
    #remove samples from unknown classes
    for i,j in reversed(removed_samples):
        del data[i][j]

    return data, numerical_annotations

def create_common_set(data, annot):
    #create bins
    N = []
    S = []
    V = []
    #F = []
    Q = []

    # from N, S and V types select randomly 75 samples each
    # add all F (13) and Q (7) types
    # resulting in 245 samples
    common = []
    labels = []
    for i, patient_sample in enumerate(data):
        for j, sample in enumerate(patient_sample):
            if (annot[i][j] == 0 ):
                N.append([i,j])
            elif (annot[i][j] == 1 ):
                S.append([i,j])
            elif (annot[i][j] == 2 ):
                V.append([i,j])
            elif (annot[i][j] == 3 ):
                common.append(sample)
                labels.append(annot[i][j])
            elif (annot[i][j] == 4 ):
                Q.append([i,j])
                common.append(sample)
                labels.append(annot[i][j])
    
    print(len(Q))

    for i in range(75):
        r = random.randint(0, len(N)-1)
        a, b = N[r]
        common.append(data[a][b])
        labels.append(annot[a][b])

        r = random.randint(0, len(S)-1)
        a, b = S[r]
        common.append(data[a][b])
        labels.append(annot[a][b])

        r = random.randint(0, len(V)-1)
        a, b = V[r]
        common.append(data[a][b])
        labels.append(annot[a][b])
    
    return common, labels
        
def print_annotations(ann):
    for patient in ann:
        print(patient)

def save_data(file, arr):
    np.save(file, arr)


def save_data(file, arr):
    path = '/'.join(file.split('/')[:-1])
    if not Path(path).exists():
        os.mkdir(path)
    np.save(file, arr)


"""def save_labels(file, arr):
    labels = []
    for patient_annot in arr:
        labels.append(patient_annot.num)
    labels = np.save(file, labels)"""

def save_labels(file, arr):
    #labels = []
    #for patient_annot in arr:
    #    labels.append(patient_annot.num)
    labels = np.save(file, arr)


def plot_data(s):
    wfdb.plot_items(s)


def main():
    training_data, training_annotations = read_data(0, 10800, all_files)
    testing_data, testing_annotations = read_data(10800, 648000, all_files)
    common_data, common_annotations = read_data(0, 648000, common_training_data)

    #print_annotations(common_annotations)
    split_common_data = process_data(common_data, common_annotations)
    common_data = reshape(split_common_data)
    common_data, common_annotations = group_to_five_classes(common_data, common_annotations)
    common_data, common_annotations = create_common_set(common_data, common_annotations)    

    split_training_data = process_data(training_data, training_annotations)
    split_testing_data = process_data(testing_data, testing_annotations, True)

    
    training_data = reshape(split_training_data)
    testing_data = reshape(split_testing_data)
    #plot_data(testing_data[24][1500])

    #training_chunks = create_three_beat_chunks(training_data)

    training_data, training_annotations = group_to_five_classes(training_data, training_annotations)
    testing_data, testing_annotations = group_to_five_classes(testing_data, testing_annotations)

    save_data('./training/X', training_data)
    save_data('./testing/X', testing_data)

    save_data('./training/common/X', common_data)
    save_data('./training/common/y', common_annotations)

    #save_data('./training/chunks/X', training_chunks)
    #save_labels('./training/chunks/y', training_annotations)

    save_labels('./training/y', training_annotations)
    save_labels('./testing/y', testing_annotations)
    

if __name__ == "__main__":
    main()
