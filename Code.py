import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import scipy.io as sio
import os
from scipy.signal import welch
from scipy.integrate import simps
import datetime
import time
import pickle




def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    band = np.asarray(band)
    low, high = band
    data = pd.DataFrame(data)
    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg = (2/ low)*sf, scaling = 'density', axis = 0)

    psd = pd.DataFrame(psd , index = freqs, columns = data.columns)

    index_min = np.argmax(np.round(freqs) > low) - 1
    index_max = np.argmax(np.round(freqs) > high)

    psd = psd.iloc[index_min:index_max,:]

    psd = psd.mean()

    return psd

def power_measure(data, freq =128):
    bandpasses = [[[8,13], 'Alpha'], [[13,30], 'Beta']]
    welch_df = pd.DataFrame()
    for bandpass, freq_name in bandpasses:
        bandpass_data= bandpower(data, freq , bandpass)
        bandpass_data.index = [freq_name]

        if welch_df.empty:
            welch_df = bandpass_data

        else:
            welch_df = pd.concat([welch_df, bandpass_data])

    welch_df = welch_df.T

    return welch_df


def stress_measure(valence, arousal):

    if (valence<= 4.5 and arousal>4.5):
        return 1
    elif(valence>4.5 and arousal>4.5):
        return 0
    else:
        return 2

Data_Location = "deap_matlab_files"
Output_Location = "ProcessedData"
Files = os.listdir(Data_Location)

os.makedirs(Output_Location, exist_ok = True)


for i, file_name in enumerate(sorted(Files)):
    print("[" + str(i + 1) + "/" + str(len(Files)) + "]", file_name)
    file_path = os.path.join(Data_Location, file_name)
    data_directory = sio.loadmat(file_path)
    print("Preprocessing file :", file_path)
    data, labels = data_directory["data"], data_directory["labels"]

    Channel_data = data[:, [1, 17, 2, 19], :]
    print("Data Shape:", Channel_data.shape)
    print("Label Shape:", labels.shape)


    Trials_dataset = pd.DataFrame()
    Total_dataset = pd.DataFrame()

    for trials in range (labels.shape[0]):
        resulta = pd.DataFrame()
        resultb = pd.DataFrame()

        result1 =  Channel_data[trials, :, :]
        result1 = pd.DataFrame(result1)
        result1 = result1.set_index(pd.Index(['AF3','AF4','F3', 'F4']))

        for i in range(len(result1)):
            row = result1.iloc[i]
            a_df = pd.DataFrame()
            b_df = pd.DataFrame()

            for i in range(63):
                f = row [i*128:(i + 1)* 128]

                band_powers = power_measure(f)
                alpha = pd.Series([band_powers[0]])
                beta = pd.Series([band_powers[1]])

                a_df = pd.concat([a_df,alpha], axis = 1, ignore_index = True)
                b_df = pd.concat([b_df,beta], axis = 1, ignore_index = True)
            
            resulta = pd.concat([resulta, a_df], axis = 0, ignore_index= True)
            resultb = pd.concat([resultb, b_df], axis = 0, ignore_index= True)

        arousal_idx = resulta.sum(axis = 0)
        arousal_den = resultb.sum(axis = 0)

        arousal  = pd.DataFrame(arousal_idx.div(arousal_den), columns = ['Arousal'])
        valence1 = (resulta.iloc[3,:]).div(resultb.iloc[3,:])
        valence2 = (resulta.iloc[2,:]).div(resultb.iloc[2,:])

        valence = pd.DataFrame(valence1.subtract(valence2, fill_value = 0), columns = ['Valence'])
        result_lf = pd.concat([valence, arousal], axis = 1)

        lab_valence = labels[trials, 0]
        lab_arousal = labels[trials, 1]

        print(arousal)
        print(valence)

        stress = stress_measure(lab_valence, lab_arousal)
        result_lf["label"] = stress

        Trials_dataset = pd.concat([Trials_dataset, result_lf], axis = 0, ignore_index = True)
    
    result_total = Trials_dataset[["Valence", "Arousal"]].to_numpy()
    result_total_label = Trials_dataset[["label"]].to_numpy()

    


from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  classification_report

import matplotlib.pyplot as plt


def load_data(processed_data_dir):
    data = {}
    for file_id in os.listdir(processed_data_dir):
        file_path = os.path.join(processed_data_dir, file_id)
        with open (file_path, "rb") as f:
            data[file_id] = pickle.load(f)
    return data


def train_test_split(data, test_split = "s01", skip_invalid = False):
    X_test = data[test_split + ".pickle"]["features"]
    Y_test = data[test_split + ".pickle"]["label"][:,0]

    X_train = []
    Y_train = []

    for split_name, data_directory in data.items():
        if test_split not in split_name:
            X_train.append(data_directory["features"])
            Y_train.append(data_directory)["label"][:,0]

    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)

    if skip_invalid:
        return X_train[Y_train!=2], Y_train[Y_train!=2], X_test[Y_test!=2], Y_test[Y_test!=2]
    return X_train,Y_train,X_test,Y_test



def visualize_features(x,y):
    plt.plot(x[:,0], "-")
    plt.plot(x[:,1], "-")
    plt.plot(np.where(y==0)[0], [0]*len(y[y==0]), "|")
    plt.plot(np.where(y==1)[0], [1]*len(y[y==1]), "|")
    plt.show()

classifier = SVC(gamma = 2, C=1)

Processed_data_directory = "ProcessedData"
skip_invalid = True

data = load_data(Processed_data_directory)

avg_test_accuracy = []
for i in range(32):
    test_split = "s" + str(i+1) if i >= 9 else "s0" +str(i+1)
    X_train, Y_train, X_test,Y_test = train_test_split(data,test_split = test_split, skip_invalid = skip_invalid)
    print("Training Data:")
    print("\tFeatures:", X_train.shape)
    print("\tLabels:", Y_train.shape)

    print("\n Test Data:", test_split)
    print("\t Features:", X_train.shape)
    print("\t Labels:", Y_test.shape)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    print("\n Training Started...")
    start_time = time.time()
    classifier.fit(X_train,Y_train)
    end_time = time.time()
    print("\n Training Time:", round (end_time - start_time,2), "seconds")
    test_accuracy = classifier.score(X_train,Y_test)
    avg_test_accuracy.append(test_accuracy)
    print("Test Accuracy:", round(test_accuracy*100,2), "%")
    Y_pred = classifier.predict(X_test)
    cls_report = classification_report(Y_test, Y_pred)
    print("\n Classification Report:\n", cls_report)
    conf_mat = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:\n", conf_mat)
    print()
avg_test_accuracy = sum(avg_test_accuracy)/len(avg_test_accuracy)
print("\n Average Test Accuracy:", round(avg_test_accuracy*100, 2), "%") 
