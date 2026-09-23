# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:30:13 2024

@author: saide
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_metrics_with_jitter(truth_matrix, output_matrix, window):
    """
    Compute metrics for spike-sorting performance considering jitter tolerance.

    Parameters:
    truth_matrix (numpy.ndarray): A t x n matrix with ground truth spikes.
    output_matrix (numpy.ndarray): A k x n matrix with predicted spikes.
    window (int): The number of timesteps to consider for jitter tolerance.

    Returns:
    dict: A dictionary containing P_D, R_D, F_D, C, and F.
    list: A list of dictionaries containing P_ij, R_ij, and F_ij for each pair (i, j).
    dict: A dictionary containing the best (i, j) pairs and their pairwise metrics.
    """
    t, n = truth_matrix.shape
    k, _ = output_matrix.shape

    # Initialize counts
    D = 0  # Correct detections
    FN = 0  # False negatives
    FP = 0  # False positives

    # Precision, Recall, F-score for each pair (i, j)
    pairwise_metrics = []

    # Compute pairwise metrics and total hits
    hits_matrix = np.zeros((t, k))

    for i in range(t):
        Ti = np.sum(truth_matrix[i])
        for j in range(k):
            Oj = np.sum(output_matrix[j])
            Hij = 0  # Number of correctly classified elements for pair (i, j)
            for col in range(n):
                if truth_matrix[i, col] == 1:
                    detected = False
                    # for w in range(max(0, col-window), min(n, col+window+1)):
                    for w in range(col, min(n, col+window+1)):
                        if output_matrix[j, w] == 1:
                            detected = True
                            break
                    if detected:
                        Hij += 1

            # Calculate P_ij, R_ij, and F_ij
            P_ij = Hij / Oj if Oj > 0 else 0
            R_ij = Hij / Ti if Ti > 0 else 0
            F_ij = (2 * Hij) / (Ti + Oj) if (Ti + Oj) > 0 else 0

            pairwise_metrics.append({
                "i": i,
                "j": j,
                "P_ij": P_ij,
                "R_ij": R_ij,
                "F_ij": F_ij
            })

            # Update hits matrix
            hits_matrix[i, j] = Hij

    # Calculate the optimal correspondence H_M using the hits_matrix and save the pairs
    H_M, best_pairs = optimal_hits(hits_matrix)
    final_matched_pairs = {i[0]:i[1] for i in list(best_pairs.keys())}

    # Compute total correct detections, false negatives, and false positives
    for i in range(t):
        Ti = np.sum(truth_matrix[i])
        FN += Ti - np.max(hits_matrix[i])

    for j in range(k):
        Oj = np.sum(output_matrix[j])
        FP += Oj - np.max(hits_matrix[:, j])

    D = np.sum(hits_matrix)

    # Compute detection metrics
    detection_metrics = compute_detection_metrics(D, FN, FP)

    # Compute clustering performance index
    C = compute_clustering_performance(H_M, D)

    # Total number of true and detected action potentials
    T = np.sum(truth_matrix)
    O = np.sum(output_matrix)

    # Compute global performance F-score
    F = compute_global_performance(detection_metrics["F_D"], C, H_M, T, O)

    # Get pairwise metrics for best pairs
    best_pairwise_metrics = [
        {
            "i": i,
            "j": j,
            "P_ij": hits_matrix[i, j] / np.sum(output_matrix[j]) if np.sum(output_matrix[j]) > 0 else 0,
            "R_ij": hits_matrix[i, j] / np.sum(truth_matrix[i]) if np.sum(truth_matrix[i]) > 0 else 0,
            "F_ij": (2 * hits_matrix[i, j]) / (np.sum(truth_matrix[i]) + np.sum(output_matrix[j])) if (np.sum(truth_matrix[i]) + np.sum(output_matrix[j])) > 0 else 0
        }
        for i, j in best_pairs.keys()
    ]

    # Return all metrics and best pairs with their pairwise metrics
    return {
        "P_D": detection_metrics["P_D"],
        "R_D": detection_metrics["R_D"],
        "F_D": detection_metrics["F_D"],
        "C": C,
        "F": F
    }, pairwise_metrics, final_matched_pairs, best_pairwise_metrics, hits_matrix

def optimal_hits(hits_matrix):
    """
    Compute the optimal number of hits H_M using the hits matrix and return the best pairs.

    Parameters:
    hits_matrix (numpy.ndarray): A t x k matrix with hits for each pair (i, j).

    Returns:
    int: The optimal number of hits H_M.
    dict: A dictionary containing the best (i, j) pairs.
    """
    # Hungarian algorithm to find the optimal assignment
    cost_matrix = -hits_matrix  # Maximize hits
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    H_M = hits_matrix[row_ind, col_ind].sum()

    best_pairs = {(row, col): hits_matrix[row, col] for row, col in zip(row_ind, col_ind)}

    return H_M, best_pairs

def compute_detection_metrics(D, FN, FP):
    """
    Compute precision (P_D), recall (R_D), and F-score (F_D) for detection performance.
    """
    P_D = D / (D + FN) if (D + FN) > 0 else 0
    R_D = D / (D + FP) if (D + FP) > 0 else 0
    F_D = (2 * D) / ((2 * D) + FP + FN) if ((2 * D) + FP + FN) > 0 else 0
    
    return {
        "P_D": P_D,
        "R_D": R_D,
        "F_D": F_D
    }

def compute_clustering_performance(H_M, D):
    """
    Compute the clustering performance index (C).
    """
    return H_M / D if D > 0 else 0

def compute_global_performance(F_D, C, H_M, T, O):
    """
    Compute the global performance F-score (F).
    """
    F = (2 * H_M) / (T + O) if (T + O) > 0 else 0
    return F

def epoch_fscore(truth_matrix, output_matrix, best_pairs, window, epochs):
    """
    Compute the F-scores for each epoch based on globally matched pairs.
    
    Parameters:
    truth_matrix (np.ndarray): Global truth matrix.
    output_matrix (np.ndarray): Global output matrix.
    best_pairs (dict): The best (i, j) pairs obtained from global matching.
    window (int): The number of timesteps to consider for jitter tolerance.
    epochs (int): Number of epochs to divide the matrices into.

    Returns:
    list: List of F-scores for each epoch based on global matching.
    """
    # Split the matrices into equal parts based on the number of epochs
    n_timesteps = truth_matrix.shape[1]
    epoch_size = n_timesteps // epochs
    
    fscores_per_epoch = []

    for epoch_idx in range(epochs):
        # Get the slice of the matrix for the current epoch
        start = epoch_idx * epoch_size
        end = (epoch_idx + 1) * epoch_size if epoch_idx < epochs - 1 else n_timesteps
        
        truth_epoch = truth_matrix[:, start:end]
        output_epoch = output_matrix[:, start:end]

        epoch_fscores = []
        

        # Compute F-scores for the best global (i, j) pairs
        for i in best_pairs:
            j = best_pairs[i]
            Ti = np.sum(truth_epoch[i])
            Oj = np.sum(output_epoch[j])
            Hij = 0  # Number of correctly classified elements for pair (i, j)

            # Loop through each timestep and count correct detections within the window
            n = truth_epoch.shape[1]
            for col in range(n):
                if truth_epoch[i, col] == 1:
                    detected = False
                    for w in range(max(0, col-window), min(n, col+window+1)):
                        if output_epoch[j, w] == 1:
                            detected = True
                            break
                    if detected:
                        Hij += 1

            # Calculate the F-score for this epoch and pair (i, j)
            P_ij = Hij / Oj if Oj > 0 else 0
            R_ij = Hij / Ti if Ti > 0 else 0
            F_ij = (2 * Hij) / (Ti + Oj) if (Ti + Oj) > 0 else 0

            epoch_fscores.append(F_ij)

        # Take the average F-score for all best pairs in this epoch
        avg_fscore = np.mean(epoch_fscores) if epoch_fscores else 0
        fscores_per_epoch.append(avg_fscore)

    return fscores_per_epoch

def sliding_window_fscore(truth_matrix, output_matrix, best_pairs, jitter_window, sliding_window, sliding_hop):
    """
    Compute F-scores for sliding windows within a single epoch's data.

    Parameters:
    truth_matrix (np.ndarray): Ground truth binary matrix for a single epoch.
    output_matrix (np.ndarray): Model's output binary matrix for a single epoch.
    best_pairs (dict): The best (i, j) pairs obtained from global matching.
    sliding_window (int): Size of the sliding window (in timesteps).
    sliding_hop (int): Hop size for moving the sliding window.
    jitter_window (int): Number of timesteps to consider for jitter tolerance.

    Returns:
    list: List of F-scores for each sliding window based on global matching.
    """
    n_timesteps = truth_matrix.shape[1]
    fscores_per_window = []

    # Iterate over sliding windows
    for start in range(0, n_timesteps - sliding_window + 1, sliding_hop):
        end = start + sliding_window
        truth_window = truth_matrix[:, start:end]
        output_window = output_matrix[:, start:end]

        window_fscores = []

        # Compute F-scores for the best global (i, j) pairs
        for i in best_pairs:
            j = best_pairs[i]
            Ti = np.sum(truth_window[i])
            Oj = np.sum(output_window[j])
            Hij = 0  # Number of correctly classified elements for pair (i, j)

            # Loop through each timestep in the sliding window
            n = truth_window.shape[1]
            for col in range(n):
                if truth_window[i, col] == 1:
                    detected = False
                    for w in range(max(0, col - jitter_window), min(n, col + jitter_window + 1)):
                        if output_window[j, w] == 1:
                            detected = True
                            break
                    if detected:
                        Hij += 1

            # Calculate the F-score for this window and pair (i, j)
            P_ij = Hij / Oj if Oj > 0 else 0
            R_ij = Hij / Ti if Ti > 0 else 0
            F_ij = (2 * Hij) / (Ti + Oj) if (Ti + Oj) > 0 else 0

            window_fscores.append(F_ij)

        # Take the average F-score for all best pairs in this window
        avg_fscore = np.mean(window_fscores) if window_fscores else 0
        fscores_per_window.append(avg_fscore)

    return fscores_per_window
