# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:08:37 2022

@author: saide
"""
from constants import ALL_CONSTANTS
import numpy as np


# Governing equations for an LTS Neuron

# LTS Neuron Voltage update rule
def update_v(q, prev_v, i, datatype):
    '''
    Inputs:
    q -      The new updated q(t+dt).
    prev_v - The neurons' V values from the previous time column; V(t).
    i -      The incoming current streams (spikes*weights)
    
    Output:
    v -      The new V values; V(t+dt).
    '''
    # Get constants according to datatype
    constants = ALL_CONSTANTS[datatype]
    
    prev_v = prev_v.reshape(constants['N_OUT'],1)
    q = q.reshape(constants['N_OUT'],1)
    i = i.reshape(constants['N_OUT'],1)
    v = (-prev_v + q + constants['G']*i)/constants['T_M']
    
    return np.asarray(v)


# 'q' values update rule
def update_q(f_v, prev_q, datatype):
    '''
    Inputs:
    f_v -     The f(v) values of the previous time window's V values; f(V(t)).
    prev_q -  The q values from the previous time column; q(t).
    
    Output:
    q -       The new q values; q(t+dt) which are then used to compute V(t+dt).
    '''
    
    # Get constants according to datatype
    constants = ALL_CONSTANTS[datatype]
    
    prev_q = prev_q.reshape(constants['N_OUT'],1)
    f_v = f_v.reshape(constants['N_OUT'],1)
    q = ((-prev_q + f_v)*constants['EPSILON'])/constants['T_M']
    
    return np.asarray(q)


# f(v) update rule
def f_v(v, datatype):
    '''
    Input:
    v -   The neurons' V values from the previous time column; V(t).
    
    Output:
    fv -  The f(V(t)) values which are then used to compute q(t+dt).
    
    '''
    # Get constants according to datatype
    constants = ALL_CONSTANTS[datatype]
    fv = np.where(v<0, constants['ALPHA_N']*v, constants['ALPHA_P']*v)
    
    return fv