# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:35:51 2022

@author: saide
"""

import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from utils import generate_artificial_data
from train import train
import numpy as np
import metric
import pandas as pd
import seaborn as sns
from constants import ALL_CONSTANTS

# Generate artificial data (no overlap) and the corresponding truth spike trains
repetitions=50
input_spikes, truth = generate_artificial_data.generate_artificial_data(0, reps=repetitions)

# Visualize the first few epochs of the input patterns
fig, ax = plt.subplots(figsize=(8, 6))

epochs = 4
index = int(len(input_spikes.T)/repetitions*epochs)
sns.heatmap(input_spikes[:,:index], ax=ax, cbar=False)

# Set the title and labels
ax.set_title('Patterns with no overlap')
ax.set_xlabel('Epochs')
ax.set_ylabel('Spike encoded input')
ax.invert_yaxis()
y_ticks = np.arange(0, len(input_spikes)+1,step=80)
y_labels = list(map(str, y_ticks))
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
x_ticks = np.linspace(0, len(input_spikes[:,:index].T), epochs+1)
x_labels = list(map(str, list(np.arange(epochs+1))))
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=0)

# Save figure 
plt.savefig('../results/figure_4a.png', dpi=300)

# Train the network on the input spikes to obtain postsynaptic output spikes produced by the network
output_spikes, weights, thresholds, voltages = train(input_spikes, datatype='artificial')
final_weights = weights[:,:,-1]
final_thresholds = thresholds[:,-1]

# Match truth spike trains and output spike trains
matched_pairs = metric.match(truth, output_spikes)

# Compute f-scores for each epoch
fscores = metric.evolving_fscore_artificial(truth, output_spikes)

# Reorder the output spike trains in the same order as the truth spike trains
order = [*matched_pairs.values()]
order = sorted(set(order), key=order.index)
order.extend(x for x, _ in enumerate(output_spikes) if x not in order)
rearranged_output_spikes = output_spikes[order,:]

# Visualize the results (truth, output, evolving fscore)
fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 4, 2]}, figsize=(8, 6), sharex=True)

# Ground truth spikes plot
spikes_df = pd.DataFrame(truth.T)
positions = spikes_df.apply(lambda x: spikes_df.index[x == 1], result_type="reduce")
colorCodes = ['C{}'.format(i) for i in range(len(truth))]
axs[0].eventplot(positions, lineoffsets=spikes_df.T.index, linelengths=0.62, linewidths = 2, colors=colorCodes)
axs[0].set_title('Patterns with no overlap')
axs[0].set_ylabel('Ground\ntruth')
y_ticks = np.arange(0, len(truth))
positive_indices = map(lambda x:x+1, list(y_ticks))
y_labels = list(map(str, positive_indices))
axs[0].set_yticks(y_ticks)
axs[0].set_yticklabels(y_labels)
axs[0].invert_yaxis()
axs[0].set_xticks([])


# Output spikes plot
spikes_df = pd.DataFrame(rearranged_output_spikes.T)
positions = spikes_df.apply(lambda x: spikes_df.index[x == 1], result_type="reduce")
colorCodes = ['C{}'.format(i) for i in range(len(rearranged_output_spikes))]
axs[1].eventplot(positions, lineoffsets=spikes_df.T.index, linelengths=0.62, linewidths = 2, colors=colorCodes)
axs[1].set_ylabel('Output spikes\n(LTS neurons)')
y_ticks = np.arange(0, len(order))
non_zero_order = map(lambda x:x+1, order)
y_labels = list(map(str, non_zero_order))
axs[1].set_yticks(y_ticks)
axs[1].set_yticklabels(y_labels)
axs[1].invert_yaxis()
axs[1].set_xticks([])

# F-score plot
axs[2].plot(fscores, color='blue')
axs[2].set_ylabel('f-score')
axs[2].set_xlabel('Epochs')
axs[2].get_shared_x_axes().remove(axs[2])
axs[2].axhline(y=1, color='g', linestyle='--')


plt.tight_layout()
plt.show()

# Save figure
plt.savefig('../results/figure_4b.png', dpi=300)


# Generate artificial data (one inside the other) and the corresponding truth spike trains
input_spikes, truth = generate_artificial_data.generate_artificial_data(1, reps=50)

# Visualize the first few epochs of the input patterns
fig, ax = plt.subplots(figsize=(8, 6))

epochs = 4
index = int(len(input_spikes.T)/repetitions*epochs)
sns.heatmap(input_spikes[:,:index], ax=ax, cbar=False)

# Set the title and labels
ax.set_title('Patterns that are one inside the other')
ax.set_xlabel('Epochs')
ax.set_ylabel('Spike encoded input')
ax.invert_yaxis()
y_ticks = np.arange(0, len(input_spikes)+1,step=80)
y_labels = list(map(str, y_ticks))
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
x_ticks = np.linspace(0, len(input_spikes[:,:index].T), epochs+1)
x_labels = list(map(str, list(np.arange(epochs+1))))
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=0)

# Save figure
plt.savefig('../results/figure_4c.png', dpi=300)

# Train the network on the input spikes to obtain postsynaptic output spikes produced by the network
output_spikes, weights, thresholds, voltages = train(input_spikes, datatype='artificial')
final_weights = weights[:,:,-1]
final_thresholds = thresholds[:,-1]

# Match the few final epochs of the output spike trains with their corresponding truth spike trains
matched_pairs = metric.match(truth[:,-index:], output_spikes[:,-index:])

# Compute fscores for each epoch
fscores = metric.evolving_fscore_artificial(truth, output_spikes)

# Reorder the output spike trains in the same order as the truth spike trains
order = [*matched_pairs.values()]
order = sorted(set(order), key=order.index)
order.extend(x for x, _ in enumerate(output_spikes) if x not in order)
rearranged_output_spikes = output_spikes[order,:]

# Visualize the results (truth, output, evolving fscore)
fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 4, 2]}, figsize=(8, 6), sharex=True)

# Ground truth spikes plot
spikes_df = pd.DataFrame(truth.T)
positions = spikes_df.apply(lambda x: spikes_df.index[x == 1], result_type="reduce")
colorCodes = ['C{}'.format(i) for i in range(len(truth))]
axs[0].eventplot(positions, lineoffsets=spikes_df.T.index, linelengths=0.62, linewidths = 2, colors=colorCodes)
axs[0].set_title('Patterns that are one inside the other')
axs[0].set_ylabel('Ground\ntruth')
y_ticks = np.arange(0, len(truth))
positive_indices = map(lambda x:x+1, list(y_ticks))
y_labels = list(map(str, positive_indices))
axs[0].set_yticks(y_ticks)
axs[0].set_yticklabels(y_labels)
axs[0].invert_yaxis()
axs[0].set_xticks([])


# Output spikes plot
spikes_df = pd.DataFrame(rearranged_output_spikes.T)
positions = spikes_df.apply(lambda x: spikes_df.index[x == 1], result_type="reduce")
colorCodes = ['C{}'.format(i) for i in range(len(rearranged_output_spikes))]
axs[1].eventplot(positions, lineoffsets=spikes_df.T.index, linelengths=0.62, linewidths = 2, colors=colorCodes)
axs[1].set_ylabel('Output spikes\n(LTS neurons)')
y_ticks = np.arange(0, len(order))
non_zero_order = map(lambda x:x+1, order)
y_labels = list(map(str, non_zero_order))
axs[1].set_yticks(y_ticks)
axs[1].set_yticklabels(y_labels)
axs[1].invert_yaxis()
axs[1].set_xticks([])

# F-score plot
axs[2].plot(fscores, color='blue')
axs[2].set_ylabel('f-score')
axs[2].set_xlabel('Epochs')
axs[2].get_shared_x_axes().remove(axs[2])
axs[2].axhline(y=1, color='g', linestyle='--')


plt.tight_layout()
plt.show()

# Save figure
plt.savefig('../results/figure_4d.png', dpi=300)