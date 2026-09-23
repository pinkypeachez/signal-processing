
import numpy as np
from tqdm import tqdm


def generate_spike_trains_robust(matrix, k, mad_arr):
    # Convert matrix to numpy array if it isn't already
    matrix = np.array(matrix)
     
    # Get dimensions of the matrix
    num_signals, n = matrix.shape
    
    # Initialize the spike trains with zeros
    spike_trains = np.zeros((2 * k * num_signals, n))
    
    # Iterate over each signal (each row in the matrix)
    for idx in tqdm(range(num_signals)):
        signal = matrix[idx]
        # Iterate over the signal starting from t = k (to handle edge cases)
        for t in range(k, n):
            for i in range(1, k + 1):
                # Compare for increase
                if (signal[t] - signal[t - i]) >= mad_arr[idx,i-1]:
                    spike_trains[idx * 2 * k + (i - 1), t] = 1
                # Compare for decrease
                if (signal[t - i] - signal[t]) >= mad_arr[idx,i-1]:
                    spike_trains[idx * 2 * k + (k + i - 1), t] = 1
    
    return spike_trains

def generate_spike_trains_robust_optimized(matrix, k, mad_arr):
    matrix = np.asarray(matrix)
    num_signals, n = matrix.shape
    spike_trains = np.zeros((2 * k * num_signals, n), dtype=np.int8)

    for idx in tqdm(range(num_signals)):
        signal = matrix[idx]

        for i in range(1, k + 1):
            # Calculate differences *once*
            diff = signal[i:] - signal[:-i]

            # Find indices where the threshold is met for increase and decrease
            increase_indices = np.where(diff >= mad_arr[idx, i - 1])[0] + i
            decrease_indices = np.where(-diff >= mad_arr[idx, i - 1])[0] + i  # Note the negation for decrease

            spike_trains[idx * 2 * k + (i - 1), increase_indices] = 1
            spike_trains[idx * 2 * k + (k + i - 1), decrease_indices] = 1

    return spike_trains






