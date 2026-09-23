# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:33:22 2024

@author: saide
"""

from encoding import sigma_delta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_neural_ss import train_neural_ss
import metric_ss
import numpy as np
from scipy.stats import median_abs_deviation
import scipy.io
import seaborn as sns
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
datapath = "./data/"
data_ss = scipy.io.loadmat(datapath+'meaExample3BY_signal.mat')['data']
truth_timestamps = scipy.io.loadmat(datapath+'meaExample3BY_truth_times.mat')['truth_timestamps'][0]
truth_neurontags = scipy.io.loadmat(datapath+'meaExample3BY_neuron_tags.mat')['truth_neuronTags'][0]

# Format truth data into spike trains
truth_spike_trains = np.zeros((max(truth_neurontags),data_ss.shape[1]))
dt = 5e-05
truth_timestamps = np.asarray(truth_timestamps/dt, dtype='int').T

for neuron,timestamp in zip(truth_neurontags, truth_timestamps):
    truth_spike_trains[neuron-1,timestamp] = 1 

# Get the first occurence of each ground truth to visualize them    
first_occurences_of_each_truth = [5010, 11536, 576, 17250, 1452, 1307]
first_occurences_of_each_truth_raw = [data_ss[:,x-50:x+50] for x in first_occurences_of_each_truth]

all_truths_array = np.hstack((first_occurences_of_each_truth_raw))

# Encode input data into spike trains
sigma_delta_k = 10

mad_list = []

for i in range(1,sigma_delta_k+1):
    differences = data_ss[:, i:] - data_ss[:, :-i]
    differences = abs(differences)
    mad = median_abs_deviation(differences, axis=1)
    mad_list.append(mad)

mad_array = np.column_stack(mad_list)

multiplier = 10
mad_array_multiplied = mad_array*multiplier
sigma_delta_multiple = sigma_delta.generate_spike_trains_robust(data_ss, sigma_delta_k, mad_array_multiplied)

# Get the first occurence of each encoded ground truth to visualize them 
first_occurences_of_each_truth_encoded = [sigma_delta_multiple[:,x-50:x+50] for x in first_occurences_of_each_truth]
all_truths_encoded_array = np.hstack((first_occurences_of_each_truth_encoded))


# Prepare for training
epochs = 10
input_spikes = np.hstack(([sigma_delta_multiple]*epochs))
truth = np.hstack(([truth_spike_trains]*epochs))

print("Training the network...\n")

output_spikes, final_weights, final_thresholds = train_neural_ss(input_spikes, datatype='neural_spikes')
results, pairwise_results, matched_pairs, matched_pairs_results, hitsmatrix = metric_ss.compute_metrics_with_jitter(truth, output_spikes, 80)

fscore_epochs = [0] + metric_ss.epoch_fscore(truth, output_spikes, matched_pairs, 80, 10)

order = [pair[1] for pair in list(matched_pairs.items())]

rearranged_output_spikes = output_spikes[order]

# Filtrered neural data before and after encoding (of each of the 6 ground truths)
fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Plotting raw neural data
for i in range(all_truths_array.shape[0]):
    axs[0].plot(all_truths_array[i] + i*20, color='tab:blue')
axs[0].set_ylabel('Electrode Channels')
axs[0].set_xlabel('Time')
axs[0].set_title('Filtered neural data')
axs[0].set_xticks([])
axs[0].set_yticks([])

# Plotting encoded spike trains
sns.heatmap(all_truths_encoded_array, ax=axs[1], cbar=False)
axs[1].set_title('Final spike trains')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Spike encoded input')
axs[1].invert_yaxis()
axs[1].set_xticks([])
axs[1].set_yticks([])

plt.tight_layout()
plt.show()

# Save figure
plt.savefig('../results/figure_2de.png', dpi=300)

# Fscore evolution across epochs
fig, axs = plt.subplots(1, 1, figsize=(8, 8))

# Plotting the fscore
axs.plot(fscore_epochs, marker='x', markersize=10, linestyle='-', linewidth=2)
axs.set_ylabel('f-score')
axs.set_xlabel('Epochs')
axs.set_xticks([])
axs.set_yticks([])

plt.tight_layout()
plt.show()

# Save figure
plt.savefig('../results/figure_7a.png', dpi=300)

# Spike train visualiation of the ground truths and their matched output for the lat epoch
fig, axes = plt.subplots(6, 1, figsize=(12, 10))
for i in range(len(truth)):
    pair = np.vstack((truth[i,-400000:], rearranged_output_spikes[i,-400000:]))
    spikes_df = pd.DataFrame(pair.T)
    positions = spikes_df.apply(lambda x: spikes_df.index[x == 1], result_type="reduce")
    colorCodes = ['C{}'.format(i) for i in range(len(pair))]
    axes[i].eventplot(positions, lineoffsets=spikes_df.T.index, linelengths=0.62, linewidths = 2, colors=colorCodes)
    axes[i].invert_yaxis()

for ax in axes:
    ax.axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

# Save figure
plt.savefig('../results/figure_7b.png', dpi=300)
