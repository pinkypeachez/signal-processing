# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:03:11 2022

@author: saide
"""
# Imports
import numpy as np
import time
from tqdm import tqdm
from lts_neuron import lts
from constants import ALL_CONSTANTS


def train_neural(input_spikes, datatype, stp_cutoff=0.75, stp_threshold=1, rows_per_channel=24, spikes_per_channel=5, residual_check=0.6):
    
    constants = ALL_CONSTANTS[datatype]
    
    output = np.zeros((constants['N_OUT'],1))
    q = np.zeros((constants['N_OUT'],1))
    # output_spikes = np.zeros((constants['N_OUT'],1))
    output_spikes = []
    threshold = constants['TH']*np.ones((constants['N_OUT'],1))
    stp_continue = True

    # Initialize weights
    lower_limit = -1
    upper_limit = 0
    
    np.random.seed(1)
    weights = np.random.uniform(low=lower_limit, high=upper_limit,size=(constants['N_OUT'],input_spikes.shape[0]))

    stp_weights = np.ones((input_spikes.shape[0]))
    
    
    # Starting timer to compute execution time.
    start = time.time()
    
    for col in tqdm(range(len(input_spikes.T)), colour='green'):
        
        currents = input_spikes[:,col].dot((stp_weights*weights).T)
        # print(currents.shape)
        
        spikes_list = list(np.where(input_spikes[:,col]==1)[0])
        # print(spikes_list)
        
        # Compute f(v) with the previous v
        f = lts.f_v(output, datatype).reshape(constants['N_OUT'],1)

        # Compute new 'q' value using previous 'q' and f(v) 
        new_q = lts.update_q(f, q, datatype).reshape(constants['N_OUT'],1)

        # q(t+dt) = q(t) + dq
        new_q = new_q + q.reshape(constants['N_OUT'],1)

        # Copmute the new 'V' value using previous 'V', new 'q' and incoming currents
        new_v = lts.update_v(q, output, currents, datatype).reshape(constants['N_OUT'],1)

        # V(t+dt) = V(t) + dV
        new_v = new_v + output.reshape(constants['N_OUT'],1)
    
               
        if stp_continue:
            
            dw = np.zeros((input_spikes.shape[0]))
            dw = (1-stp_weights)/constants['T_STP']
            
            for freq in spikes_list:
                dw[freq] = -(stp_weights[freq]*constants['FD'])

            stp_weights += dw
            stp_weights = np.clip(stp_weights, 0, 1)
            
        if stp_continue:
            if any(stp_weights<stp_cutoff):

                stp_weights = np.where(stp_weights<stp_threshold, 0, 1)
                stp_continue = False
                
                new_q = np.zeros((constants['N_OUT'],1))
                new_v = np.zeros((constants['N_OUT'],1))
                
                input_spikes = (input_spikes.T*stp_weights).T
    
                for i in range(0,len(input_spikes),int(len(input_spikes)/rows_per_channel)):
    
                    residual_cutoff = int(residual_check*spikes_per_channel)
                    residual_noise = np.where(np.sum(input_spikes[i:i+rows_per_channel,:], axis=0)<=residual_cutoff)
                    input_spikes[i:i+rows_per_channel, residual_noise] = 0
        
        # Check if threshold has been reached by any of the output neurons
        if any(new_v/threshold >= 1) and not stp_continue:
            
            threshold_check = ((new_v/threshold>=1).reshape((constants['N_OUT'],1)))
            spikes = (new_v - output.reshape(constants['N_OUT'],1)).reshape((constants['N_OUT'],1))
            spikes = spikes*threshold_check
            
            # Implement Winner-Take-All(WTA) mechanism 
            if constants['WTA']:
                
                spikes = np.where(spikes == spikes.max(keepdims=True), 1, 0)
                spike_index = np.where(spikes==1)[0][0]
                
                
                # Classical STDP
                if constants['CLASSICAL_STDP']:            
                    
                    pre_synaptic_frequencies = []
                    
                    
                    for i in range(constants['STDP_WINDOW']):
                        try:
                            pre_frequecies_stdp = np.where(input_spikes[:,col - (i + 1)]==1)[0]
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
                            pre_frequecies_lstdp = np.where(input_spikes[:,col - (i + 1)]==1)[0]
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
                            pre_frequecies_th = np.where(input_spikes[:,col - (k + 1)]==1)[0]
                            for pre_freq_th in pre_frequecies_th:
                                threshold[spike_index] += abs(weights[spike_index, pre_freq_th])*constants['TH_PAIR']               
                        except IndexError:
                            continue
                    

                        
                # Clip the weights to the interval (-1,0)
                weights = np.clip(weights, lower_limit, upper_limit)
                
                # Clip the threshold 
                np.clip(threshold, constants['TH_MIN'], constants['TH_MAX'], out=threshold)
            
            
            # Append new spikes to output spikes matrix
            output_spikes.append(spikes)
            
            # Clip the weights to the interval (-1,0)
            weights = np.clip(weights, lower_limit, upper_limit)

            
            
            # Update 'q' and 'V' matrices
            q = new_q
            output = new_v
            
            # Reset 'q' and 'V' to zero 
            new_q = np.zeros((constants['N_OUT'],1))
            new_v = np.zeros((constants['N_OUT'],1))
            
                        
            # Update 'q' and 'V' matrices
            q = new_q
            output = new_v
            
                
        else:
            
            # Append zeros column to output spikes matrix as there is no spike (threshold not reached)
            output_spikes.append(np.zeros((constants['N_OUT'],1)))

            # weights.clip(lower_limit, upper_limit, out=weights)
            weights = np.clip(weights, lower_limit, upper_limit)
            
            # Clip the thrshold 
            np.clip(threshold, constants['TH_MIN'], constants['TH_MAX'], out=threshold)
            
            # Update 'q' and 'V' matrices
            q = new_q
            output = new_v
            
            
    
        # Clip the weights to the interval (-1,0)
        weights = np.clip(weights, lower_limit, upper_limit)

        

    # Ending timer to measure execution time.
    end = time.time()
    

    print('The total time taken is {} seconds.'.format(end-start))
    print('The total number of spikes generated by all neurons is {}.'.format(np.sum(output_spikes)))
    
    
    return np.hstack(output_spikes), weights, threshold, input_spikes