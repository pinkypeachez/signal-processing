# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:36:05 2023

@author: saide
"""

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io
import metric

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing

from utils import pad_spikes, encode_neural_data
from encoding import generate_spikes_from_mels
from train_neural import train_neural

# Suppress warnings
warnings.filterwarnings('ignore')


data_path = "../data/"

print("\nLoading data...")
bloc = scipy.io.loadmat(data_path+"070710b12_bp200-2000Hz.mat")
print("\nData loaded!")

print("\nEncoding data as spikes...")
bloc_encoded = encode_neural_data.encode(bloc)
print("\nData encoded as spikes!")

del bloc

#%%
scaler = preprocessing.MinMaxScaler()
bloc_normalized = scaler.fit_transform(bloc_encoded.T)
bloc_normalized = bloc_normalized.T

#%%

receptive_fields = 20
padding = 2

print("\nGenerating input spike trains from loaded data...")
encoded_spikes = generate_spikes_from_mels.generate_spikes_from_mels(bloc_normalized, receptive_fields, padding)
final_encoded_spikes = pad_spikes.pad_spikes(encoded_spikes, padding)
print("Input spike trains generated! \n")

#%%

labels = pd.read_excel(data_path+"/070710b12_labels.xlsx")
labels = np.asarray(labels*100, dtype='int').T
long_labels = labels[0]
short_labels = labels[1][:3]

truth_spike_trains = np.zeros((len(labels), final_encoded_spikes.shape[1]))

for long in long_labels:
    truth_spike_trains[1, long] = 1

for short in short_labels:
    truth_spike_trains[0, short] = 1

#%%
epochs = 10
input_spikes = np.hstack(([final_encoded_spikes]*epochs))
truth = np.hstack(([truth_spike_trains]*epochs))

print("Training the network...\n")
output_spikes, final_weights, final_thresholds, input_spikes_after_stp = train_neural(input_spikes, datatype='neural')

#%%

# Choose start and end indices of a random interval of the neural data containing one short pattern and one long pattern
start = 191000
end = 207000

# Plotting the layout
fig, axs = plt.subplots(3, 1, figsize=(6, 10), gridspec_kw={'height_ratios': [1, 1, 1]})

# Top subplot
for ch in range(bloc_normalized[:,start:end].shape[0]):
    axs[0].plot(bloc_normalized[ch,start:end] + ch, color='tab:blue')
axs[0].set_title('Spike envelope')
axs[0].set_xlabel('Time(s)')
axs[0].set_ylabel('Channels')
y_ticks = np.arange(0, len(bloc_normalized)+1, 6)
y_labels = list(map(str, y_ticks))
axs[0].set_yticks(y_ticks)
axs[0].set_yticklabels(y_labels)
x_ticks = np.linspace(0, len(bloc_normalized[:,start:end].T), 6)
axs[0].set_xticks(x_ticks)
x_labels = np.arange(0, len(bloc_normalized[:,start:end].T)//100, step=30)
x_labels = x_labels = list(map(str, x_labels))
axs[0].set_xticklabels(x_labels)

# Middle subplot - Heatmap
sns.heatmap(final_encoded_spikes[:,start:end], ax=axs[1], cbar=False)
axs[1].set_title('Encoded spike envelope of neural data')
axs[1].set_xlabel('Time(s)')
axs[1].set_ylabel('Spike encoded input')
axs[1].invert_yaxis()
y_ticks = np.arange(0, len(final_encoded_spikes)+1, step=120)
y_labels = list(map(str, y_ticks))
axs[1].set_yticks(y_ticks)
axs[1].set_yticklabels(y_labels)
x_ticks = np.linspace(0, len(final_encoded_spikes[:,start:end].T), 6)
axs[1].set_xticks(x_ticks)
x_labels = np.arange(0, len(final_encoded_spikes[:,start:end].T)//100, step=30)
x_labels = x_labels = list(map(str, x_labels))
axs[1].set_xticklabels(x_labels, rotation=0)


# Bottom subplot - Heatmap
sns.heatmap(input_spikes_after_stp[:,start:end], ax=axs[2], cbar=False)
axs[2].set_title('Final spike trains after STP')
axs[2].set_xlabel('Time(s)')
axs[2].set_ylabel('Spike encoded input')
axs[2].invert_yaxis()
y_ticks = np.arange(0, len(input_spikes_after_stp)+1, step=120)
y_labels = list(map(str, y_ticks))
axs[2].set_yticks(y_ticks)
axs[2].set_yticklabels(y_labels)
x_ticks = np.linspace(0, len(input_spikes_after_stp[:,start:end].T), 6)
axs[2].set_xticks(x_ticks)
x_labels = np.arange(0, len(input_spikes_after_stp[:,start:end].T)//100, step=30)
x_labels = x_labels = list(map(str, x_labels))
axs[2].set_xticklabels(x_labels, rotation=0)


plt.tight_layout()
plt.show()

# Save figure
plt.savefig('../results/figure_1dfh.png', dpi=300)

#%%

fscores = metric.evolving_fscore_neural(truth, output_spikes, gaussian_size=3001)

# Plotting the layout
fig, axs = plt.subplots(3, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 2, 1]})

# Top subplot - Ground truth event plot
spikes_df = pd.DataFrame(truth.T)
positions = spikes_df.apply(lambda x: spikes_df.index[x == 1], result_type="reduce")
axs[0].eventplot(positions, lineoffsets=spikes_df.T.index, linelengths=0.62, linewidths = 2, colors=['tab:blue', 'tab:orange'])
axs[0].set_title('Ground truth v/s output')
axs[0].set_ylabel('Ground truth')
axs[0].set_yticks([0, 1])
axs[0].set_yticklabels(['short', 'long'])
axs[0].invert_yaxis()
axs[0].set_xticks([])


# Middle subplot - Output spikes event plot
spikes_df = pd.DataFrame(output_spikes.T)
positions = spikes_df.apply(lambda x: spikes_df.index[x == 1], result_type="reduce")
axs[1].eventplot(positions, lineoffsets=spikes_df.T.index, linelengths=0.62, linewidths = 2, colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:orange', 'tab:brown'])
axs[1].set_yticks(np.arange(0, 5))
axs[1].set_yticklabels(np.arange(1, 6))
axs[1].invert_yaxis()
axs[1].set_ylabel('Output spikes\n(LTS neurons)')
axs[1].set_xticks([])

# Bottom subplot - f-score plot
axs[2].plot(fscores, color='tab:blue')
axs[2].axhline(y=1, color='g', linestyle='--')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('f-score')
axs[2].set_ylim(0, 1.1)
axs[2].set_yticks([0, 1])
axs[2].set_yticklabels(['0', '1'])
x_ticks = np.linspace(0, len(fscores), epochs+1)
axs[2].set_xticks(x_ticks)
x_labels = list(np.arange(1,epochs+1)) + ['']
x_labels = list(map(str, x_labels))
axs[2].set_xticklabels(x_labels)

plt.tight_layout()
plt.show()

# Save figure
plt.savefig('../results/figure_6a.png', dpi=300)

#%%

# Plotting the heatmap
plt.figure(figsize=(8, 10))
ax = sns.heatmap(final_weights.T, cmap='Blues_r', cbar=True)

# Customizing the plot
ax.set_title('Final learned weights')
ax.set_xlabel('LTS neurons')
ax.set_ylabel('Spike encoded input')
ax.invert_yaxis()
y_ticks = np.arange(0, len(final_weights.T)+1, step=180)
y_labels = list(map(str, y_ticks))
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set_xticks(np.arange(len(final_weights)) + 0.5)
ax.set_xticklabels(np.arange(1, len(final_weights) + 1))

plt.tight_layout()
plt.show()

# Save figure
plt.savefig('../results/figure_6b.png', dpi=300)