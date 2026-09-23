# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:13:44 2023

@author: saide
"""

import numpy as np
from math import pi, sqrt, exp
import matplotlib.pyplot as plt
import matplotlib
import random 

def match(truth, output, gaussian_size=31):
    '''

    Parameters
    ----------
    truth : array
        Truth spike trains.
    output : array
        Output spike trains produced by the network.

    Returns
    -------
    dictionary
        Dictionary containing matched paris of patterns and neurons.

    '''
        
    def gauss(n,sigma):
        r = range(-int(n/2),int(n/2)+1)
        return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    
    gaussian = gauss(gaussian_size,3)
    
    truth_signal = np.copy(truth)
    output_signal = np.copy(output)

    for i in range(len(truth)):
        truth_signal[i] = np.convolve(truth[i],gaussian,mode='same')
    for a in range(len(output)):
        output_signal[a] = np.convolve(output[a],gaussian,mode='same')
        
    pairs = {}
    
    for i in range(len(truth_signal)):
        corr = []
        for j in range(len(output_signal)):
            corr.append(np.correlate(truth_signal[i],output_signal[j]))
        pairs[i] = np.argmax(corr)
        # print(corr)

    
    return(pairs)

def confusion_matrix(truth, output, window=40, gaussian_size=31):
    
    f_score_per_pair = []
    accuracy_per_pair = []
    global_hits = 0

    pairs_to = match(truth, output, gaussian_size=gaussian_size)
    
    confusion_matrix = np.zeros((len(truth), len(truth)))
    
    # For each true label, check if there is an output spike that corresponds
    for key in pairs_to:
        h = 0
        spikes = list(np.where(truth[key]==1)[0])
        for spike in spikes:
            if np.sum(output[pairs_to[key],spike-window:spike+window])==1:
                h += 1
                global_hits += 1
                confusion_matrix[key,key] += 1 
            else:
                wrong = np.where(output[:,spike-window:spike+window]==1)[0][0]
                confusion_matrix[key, wrong] += 1
                
        f_score_per_pair.append((2*h)/(len(spikes)+sum(output[pairs_to[key],:])))
        accuracy_per_pair.append(h/len(spikes))
    
    confusion_matrix_percentages = np.zeros((len(truth),len(truth)))
    
    sums = np.sum(confusion_matrix, axis=1)
    confusion_matrix_percentages = np.round(confusion_matrix / sums[:, np.newaxis], decimals=2)
        
    confusion_matrix_percentages = confusion_matrix_percentages*100
         
    accuracy_per_pair = [ '%.2f' % elem for elem in accuracy_per_pair]    
    
    plot = random.randint(1,1000)
    plt.figure(plot)
    
    plt.imshow(confusion_matrix_percentages, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['a','i', 'ou', 'u', 'é', 'è', 'eu', 'o', 'an', 'on', 'in']
    plt.title('Vowels classification (%)')
    plt.ylabel('True label', labelpad=44)
    plt.xlabel('Predicted label', labelpad=44)
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=0)
    plt.yticks(tick_marks, classNames)
    for i in range(len(classNames)):
        for j in range(len(classNames)):
            plt.text(j,i,'%.0f' % confusion_matrix_percentages[i][j],ha='center',va='center',color='white' if confusion_matrix[i, j] == 0 else "black")
    
    plt.tight_layout()
    plt.show()
    
    

def compute_fscore(truth, output, window=40, gaussian_size=31, pairs={}):
    '''
    

    Parameters
    ----------
    truth : array
        Truth spike trains.
    output : array
        Output spike trains produced by the network.
    window : int, optional
        Number of time-steps to adjust for jitter when matching spike trains. The default is 40.

    Returns
    -------
    Computed f-score.

    '''
    
    f_score_per_pair = []
    global_hits = 0
    
    if not pairs:
        pairs = match(truth, output, gaussian_size=gaussian_size)    
    
    # For each true label, check if there is an output spike that corresponds
    for key in pairs:
        h = 0
        spikes = list(np.where(truth[key]==1)[0])
        for spike in spikes:
            if np.sum(output[pairs[key],spike-window:spike+window])==1:
                h += 1
                global_hits += 1
                
        f_score_per_pair.append((2*h)/(len(spikes)+sum(output[pairs[key],:])))
    
    f_score_per_pair = [ '%.2f' % elem for elem in f_score_per_pair]    
    global_fscore = (2*global_hits)/(np.sum(truth)+np.sum(output))
    
    return f_score_per_pair, global_fscore
    
    
def compute_accuracy(truth, output, window=30, gaussian_size=31, pairs={}):
    '''
    

    Parameters
    ----------
    truth : array
        Truth spike trains.
    output : array
        Output spike trains produced by the network.
    window : int, optional
        Number of time-steps to adjust for jitter when matching spike trains. The default is 40.

    Returns
    -------
    Computed f-score.

    '''

    
    accuracy_per_pair = []
    global_hits = 0

    pairs_to = match(truth, output, gaussian_size=gaussian_size)
    
    
    # For each true label, check if there is an output spike that corresponds
    for key in pairs_to:
        h = 0
        spikes = list(np.where(truth[key]==1)[0])
        for spike in spikes:
            if np.sum(output[pairs_to[key],spike-window:spike+window])==1:
                h += 1
                global_hits += 1
                
        accuracy_per_pair.append((h)/(len(spikes)))
    
    accuracy_per_pair = [ '%.2f' % elem for elem in accuracy_per_pair]    
    global_accuracy = (global_hits)/(np.sum(truth))
    
    return accuracy_per_pair, global_accuracy
    
    
def evolving_fscore_neural(truth, output, window=332717, hop=332717, gaussian_size=31, title="Evolving f-score"):
    '''

    Parameters
    ----------
    truth : array
        Truth spike trains.
    output : TYPE
        Output spike trains.
    window : int, optional
        window size to compute f-score metric. The default is 6000.
    hop : int, optional
        window slide length. The default is 1200.

    Returns
    -------
    list_fscores : list
        list of all f-scores computed.

    '''
    
    list_fscores = [0]
    truth_output_pairs = match(truth[:,-500000:], output[:,-500000:], gaussian_size=gaussian_size)
    
    for index in range(0, len(truth.T), hop):
        _, fscore = compute_fscore(truth[:,index:index+window], output[:,index:index+window], window=3000, gaussian_size=gaussian_size, pairs=truth_output_pairs)
        list_fscores.append(fscore)
        
    
    return list_fscores

        
    
    return list_fscores

def evolving_fscore_audio(truth, output, window=58936, hop=58936, gaussian_size=31, title="Evolving f-score"):
    '''

    Parameters
    ----------
    truth : array
        Truth spike trains.
    output : TYPE
        Output spike trains.
    window : int, optional
        window size to compute f-score metric. The default is 6000.
    hop : int, optional
        window slide length. The default is 1200.

    Returns
    -------
    list_fscores : list
        list of all f-scores computed.

    '''
    
    list_fscores = [0]
    truth_output_pairs = match(truth, output, gaussian_size=gaussian_size)
    
    for index in range(0, len(truth.T), hop):
        _, fscore = compute_fscore(truth[:,index:index+window], output[:,index:index+window], window=40, gaussian_size=gaussian_size, pairs=truth_output_pairs)
        list_fscores.append(fscore)
        
    
    return list_fscores
    
        
def evolving_fscore_artificial(truth, output, window=360, hop=360, gaussian_size=31, title="Evolving f-score"):
    '''

    Parameters
    ----------
    truth : array
        Truth spike trains.
    output : TYPE
        Output spike trains.
    window : int, optional
        window size to compute f-score metric. The default is 6000.
    hop : int, optional
        window slide length. The default is 1200.

    Returns
    -------
    list_fscores : list
        list of all f-scores computed.

    '''
    
    list_fscores = [0]
    truth_output_pairs = match(truth[:,-1000:], output[:,-1000:], gaussian_size=gaussian_size)

    
    for index in range(0, len(truth.T), hop):
        _, fscore = compute_fscore(truth[:,index:index+window], output[:,index:index+window], pairs=truth_output_pairs)
        list_fscores.append(fscore)

    
    return list_fscores        
    
    
    
    