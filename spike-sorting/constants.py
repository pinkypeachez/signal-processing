# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:32:15 2022

@author: saide

Defining the network constants
"""


# Defining the Network constants for different data types

ALL_CONSTANTS = {
    'artificial': {
        'N_OUT': 10,
        'EPSILON': 0.001,
        'ALPHA_N': -200,
        'ALPHA_P': -10,
        'TH_MIN': 20,
        'TH_MAX': 3500,
        'TH': 20,
        'F_TH_POST': 0.01,
        'TH_PAIR': 0.1 * 0.01,
        'T_M': 15,
        'G': 100,
        'WTA': True,
        'CLASSICAL_STDP': True,
        'LATERAL_STDP': True,
        'INTRINSIC_PLASTICITY': True,
        'LEARNING_RATE': 0.5,
        'STDP_WINDOW': 50,
        'IP_WINDOW': 50
    },
    'audio': {
        'N_OUT': 11,
        'EPSILON': 0.01,
        'ALPHA_N': -200,
        'ALPHA_P': -10,
        'TH_MIN': 20,
        'TH_MAX': 3500,
        'TH': 20,
        'F_TH_POST': 0.01,
        'TH_PAIR': 0.6 * 0.01,
        'T_M': 15,
        'G': 100,
        'WTA': True,
        'STP': True,
        'T_STP': 2000,
        'FD': 0.003,
        'CLASSICAL_STDP': True,
        'LATERAL_STDP': True,
        'INTRINSIC_PLASTICITY': True,
        'LEARNING_RATE': 0.5,
        'STDP_WINDOW': 50,
        'IP_WINDOW': 50
    },
    'neural': {
        'N_OUT': 5,
        'EPSILON': 0.001,
        'ALPHA_N': -200,
        'ALPHA_P': -10,
        'TH_MIN': 20,
        'TH_MAX': 3500,
        'TH': 20,
        'F_TH_POST': 0.5,
        'TH_PAIR': 0.1 * 0.5,
        'T_M': 50,
        'G': 100,
        'WTA': True,
        'STP': True,
        'T_STP': 2000,
        'FD': 0.003,
        'CLASSICAL_STDP': True,
        'LATERAL_STDP': True,
        'INTRINSIC_PLASTICITY': True,
        'LEARNING_RATE': 0.5,
        'STDP_WINDOW': 800,
        'IP_WINDOW': 800
    },
    'neural_spikes': {
        'N_OUT': 10,
        'EPSILON': 0.001,
        'ALPHA_N': -200,
        'ALPHA_P': -10,
        'TH_MIN': 20,
        'TH_MAX': 3500,
        'TH': 20,
        'F_TH_POST': 0.5,
        'TH_PAIR': 0.1 * 0.5,
        'T_M': 15,
        'G': 100,
        'WTA': True,
        'STP': True,
        'T_STP': 2000,
        'FD': 0.003,
        'CLASSICAL_STDP': True,
        'LATERAL_STDP': True,
        'INTRINSIC_PLASTICITY': True,
        'LEARNING_RATE': 0.5,
        'STDP_WINDOW': 70,
        'IP_WINDOW': 70
    }
}



