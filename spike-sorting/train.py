# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:16:54 2024

@author: saide
"""

# Imports
import numpy as np
import time
from tqdm import tqdm
from lts_neuron import lts_neuron as lts
from constants import ALL_CONSTANTS
import copy


def train(input_spikes, datatype):
    
    # Get constants according to datatype
    constants = ALL_CONSTANTS[datatype]
    
    # Intitalize the output
    output = np.zeros((constants['N_OUT'],1))
    q = np.zeros((constants['N_OUT'],1))
    output_spikes = np.zeros((constants['N_OUT'],1))
    threshold = constants['TH']*np.ones((constants['N_OUT'],1))
    weights_list = []
    threshold_hist = []
    currents_list = []

    # Initialize weights
    lower_limit = -1
    upper_limit = 0
    
    np.random.seed(1)
    weights = np.random.uniform(low=lower_limit, high=upper_limit,size=(constants['N_OUT'],input_spikes.shape[0]))

    
    # Starting timer to compute execution time.
    start = time.time()
    
    for timestep in tqdm(range(len(input_spikes.T)), colour='green'):
        
        currents = input_spikes[:,timestep].dot(weights.T)
        
        spikes_list = list(np.where(input_spikes[:,timestep]==1)[0])
        
        # Compute f(v) with the previous v
        f = lts.f_v(output[:,-1], datatype).reshape(constants['N_OUT'],1)
        
        # Compute new 'q' value using previous 'q' and f(v) 
        new_q = lts.update_q(f, q[:,-1], datatype).reshape(constants['N_OUT'],1)
        
        # q(t+dt) = q(t) + dq
        new_q = new_q + q[:,-1].reshape(constants['N_OUT'],1)
        
        # Copmute the new 'V' value using previous 'V', new 'q' and incoming currents
        new_v = lts.update_v(q[:,-1], output[:,-1], currents, datatype).reshape(constants['N_OUT'],1)
        
        # V(t+dt) = V(t) + dV
        new_v = new_v + output[:,-1].reshape(constants['N_OUT'],1)
    
        
        # Check if threshold has been reached by any of the output neurons
        if any(new_v/threshold >= 1):
            
            # A boolean array; True where neurons have crossed their respective threshold
            threshold_check = ((new_v/threshold>=1).reshape((constants['N_OUT'],1)))
            
            # Array containing the delta (change in potential) values for each output neuron
            spikes = (new_v - output[:,-1].reshape(constants['N_OUT'],1)).reshape((constants['N_OUT'],1))
            
            # Array containing the delta values of only those neurons that have crossed their respective threshold
            spikes = spikes*threshold_check
            
            # Implement Winner-Take-All(WTA) mechanism 
            if constants['WTA']:
                
                # Choose the neuron with the highest delta (steepest rebound) as the postsynaptic spike neuron
                spikes = np.where(spikes == spikes.max(keepdims=True), 1, 0)
                spike_index = np.where(spikes==1)[0][0]
                

                
                # Classical STDP
                if constants['CLASSICAL_STDP']:            
                    
                    pre_synaptic_frequencies = []
                    
                    for i in range(constants['STDP_WINDOW']):
                        try:
                            pre_frequecies_stdp = np.where(input_spikes[:,output_spikes.shape[1] - (i + 1)]==1)[0]
                            pre_synaptic_frequencies.extend(pre_frequecies_stdp)
                        except IndexError:
                            continue
                    
                    pre_synaptic_frequencies = list(set(pre_synaptic_frequencies))
                    weights[spike_index] += 0.06*constants['LEARNING_RATE']
                    for pre_synaptic_frequency in pre_synaptic_frequencies:
                        weights[spike_index, pre_synaptic_frequency] -= 0.16*constants['LEARNING_RATE']        
                    
                
                # Lateral STDP
                if constants['LATERAL_STDP']:
                    
                    pre_synaptic_frequencies = []
                    
                    
                    for i in range(constants['STDP_WINDOW']):
                        try:
                            pre_frequecies_lstdp = np.where(input_spikes[:,output_spikes.shape[1] - (i + 1)]==1)[0]
                            pre_synaptic_frequencies.extend(pre_frequecies_lstdp)
                        except IndexError:
                            continue
                    
                    pre_synaptic_frequencies = list(set(pre_synaptic_frequencies))
                    for pre_synaptic_frequency in pre_synaptic_frequencies:
                        weights[:, pre_synaptic_frequency] += 0.0002*constants['LEARNING_RATE']
                        weights[spike_index, pre_synaptic_frequency] -= 0.0012*constants['LEARNING_RATE']
                
                # Intrinsic Plasticity
                if constants['INTRINSIC_PLASTICITY']:
                
                    
                    threshold[spike_index] -= constants['F_TH_POST']*threshold[spike_index]
                                        
                    for k in range(constants['IP_WINDOW']):
                        try:
                            pre_frequecies_th = np.where(input_spikes[:,output_spikes.shape[1] - (k + 1)]==1)[0]
                            for pre_freq_th in pre_frequecies_th:
                                threshold[spike_index] += abs(weights[spike_index, pre_freq_th])*constants['TH_PAIR']               
                        except IndexError:
                            continue
                    

                        
                # Clip the weights to the interval (-1,0)
                weights = np.clip(weights, lower_limit, upper_limit)
                
                # Clip the threshold 
                threshold = np.clip(threshold, constants['TH_MIN'], constants['TH_MAX'])
            
            # Append new spikes to output spikes matrix
            output_spikes = np.hstack((output_spikes, spikes))
            
            # Clip the weights to the interval (-1,0)
            weights = np.clip(weights, lower_limit, upper_limit)
            
            # Update 'q' and 'V' matrices
            q = np.hstack((q,new_q))
            output = np.hstack((output, new_v))
            
            # Reset 'q' and 'V' to zero 
            new_q = np.zeros((constants['N_OUT'],1))
            new_v = np.zeros((constants['N_OUT'],1))
            
            # Update 'q' and 'V' matrices
            q = np.hstack((q,new_q))
            output = np.hstack((output, new_v))
                
        else:
            
            # Append zeros column to output spikes matrix as there is no spike (threshold not reached)
            output_spikes = np.hstack((output_spikes, np.zeros((constants['N_OUT'],1))))
            
            weights = np.clip(weights, lower_limit, upper_limit)
            
            # Clip the thrshold 
            threshold = np.clip(threshold, constants['TH_MIN'], constants['TH_MAX'])
            
            # Update 'q' and 'V' matrices
            q = np.hstack((q,new_q))
            output = np.hstack((output, new_v))
    
        # Clip the weights to the interval (-1,0)
        weights = np.clip(weights, lower_limit, upper_limit) 
        
        # Store weights at each step in a list
        weights_list.append(weights)
        
        # Store threholds at each step in a list
        threshold_hist.append(threshold)
        
        # Store threholds at each step in a list
        currents_list.append(currents)    
        
            
        
    # Ending timer to measure execution time.
    end = time.time()
    
    output_spikes = output_spikes[:,:-1]
    
    # dStack the weights
    weights_stack = np.dstack(weights_list)
    threshold_stack = np.hstack(threshold_hist)
    
    print('The total time taken is {} seconds.'.format(end-start))
    print('The total number of spikes generated by all neurons is {}.'.format(np.sum(output_spikes)))
    
    
    return output_spikes, weights_stack, threshold_stack, output
