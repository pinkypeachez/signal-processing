import numpy as np
import pandas as pd
import sys, os
import ast

# Visualisierung:
import matplotlib.pyplot as plt

#Signalverarbeitung
import scipy.signal as signal
import scipy.fft

# fuer ['LINKS'] / ['RECHTS'] Klasse:
# 1. indizes fuer CLASS-START CLASS-END
# 2. Timestamps (ts) <--- werden zurueckgegeben
def extract_mi_ts(classname, df_marker):
    print("Positionen der MI-LINKS / MI-RECHTS in marker.csv werden extrahiert... ")

    i_start = np.array((df_marker.index[df_marker.iloc[:,1] == classname]))
    idx = np.append(i_start, i_start+2)
    idx = np.sort(idx)
    idx = idx.reshape((int(idx.shape[0]/2), 2)) #2 weil je 2 Werte beschreiben eindeutig eine Epoche (start, end)

    ts = []
    for i in range(len(idx)):
        #print(df_marker.iloc[idx_left[i,:], 0])
        ts.append([df_marker.iloc[idx[i,:], 0]])
    return np.array(ts)

def cut_epochs(ts, df_eeg, indizes):
    print("Epochen werden ausgeschnitten...")
    class_epochs = []
    for i in range(len(ts)):
        idx = indizes.get_indexer(ts[i,:].flatten(), method="nearest")
        epochs = df_eeg.iloc[idx,0]
        el = epochs.index.tolist()
        class_epochs.append((df_eeg.iloc[el[0]:el[1],1]).tolist())

    return (list(map(lambda el: ast.literal_eval(el),(class_epochs[0]))))

# --------------------------------------------- MAIN -------------------------------------------
# ----------- EEG und MARKER Werte in Pandas Dataframe laden
path = "../data/mi-experiment/gel-goodddd"
eeg, marker = os.listdir(path)

df_eeg = pd.read_csv(f"{path}/{eeg}", header=None)
df_marker = pd.read_csv(f"{path}/{marker}", header=None)

#----------- EEG Werte VOR Beginn des Experiments abschneiden
exp_start = df_marker.iloc[0,0]
print("[marker.csv] Start des Experiments: ", exp_start)

idx = df_eeg[df_eeg.iloc[:, 0] > exp_start].first_valid_index()
print("[eeg.csv] Start des Experiments am Index: ", idx)
print(f"[eeg.csv] Data vor {idx} wird verworfen.")
df_eeg = df_eeg[idx:] # alte index struktur wird beibehalten
df_eeg = df_eeg.reset_index(drop=True)

# ----------- Extrahiere die Timestamps von Anfang-Ende jedes Trials (fuer Linke und Rechte Klasse)
name_left = "['LINKS']"
name_right = "['RECHTS']"
ts_left = extract_mi_ts(name_left, df_marker)
ts_right = extract_mi_ts(name_right, df_marker)

# ----------- Schneide die Epochen aus eeg.csv aus
indizes = pd.Index(df_eeg.iloc[:,0])
epochs_left = cut_epochs(ts_left, df_eeg, indizes)
epochs_right = cut_epochs(ts_right, df_eeg, indizes)

print(f"Anzahl Datenpunkte: {len(epochs_left)}")


#####---------------------------------------- VISUALISIERUNG ------------------------------------------######