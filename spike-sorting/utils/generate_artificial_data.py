# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:22:29 2023

@author: saide
"""
import numpy as np


def generate_artificial_data(choice, reps=20):
    
    
    # Generate artificial spike trains

    patterns = {}
    repetitions = reps
    
    # Type 1: Four non-overlapping patterns repeating
    gap = np.zeros((240,30))
    
    voc = np.ones((60,60)) 
    v1 = np.zeros((60,60))
    v2 = np.zeros((60,60))
    v3 = np.zeros((60,60))
    
    
    voc1 = np.vstack((voc,v1,v2,v3))
    voc2 = np.vstack((v1,voc,v2,v3))
    voc3 = np.vstack((v1,v2,voc,v3))
    voc4 = np.vstack((v1,v2,v3,voc))
    
    vocs = [voc1, voc2, voc3, voc4]
    
    basic = np.hstack((voc1,gap,voc2,gap,voc3,gap,voc4,gap))
    truth_basic = np.zeros((len(vocs), len(basic.T)))
    truth_basic[0,59] = 1
    truth_basic[1,149] = 1
    truth_basic[2,239] = 1
    truth_basic[3,329] = 1
    basic = np.hstack(([basic]*repetitions))
    truth_basic = np.hstack(([truth_basic]*repetitions))
    
    patterns[0] = [basic, truth_basic]
    
    # Type 2: Four patters where two patterns are inside the other two patterns
    
    gap = np.zeros((240,30))

    big = np.ones((120,60)) 
    small = np.ones((60,60))
    fill_big = np.zeros((120,60))
    fill_small = np.zeros((180,60))
    
    
    voc1 = np.vstack((small,fill_small))
    voc2 = np.vstack((big, fill_big))
    voc3 = np.vstack((fill_small, small))
    voc4 = np.vstack((fill_big, big))
    
    vocs = [voc1, voc2, voc3, voc4]
    
    one_inside_other = np.hstack((voc1,gap,voc2,gap,voc3,gap,voc4,gap))
    truth_one_inside_other = np.zeros((len(vocs),len(one_inside_other.T)))
    truth_one_inside_other[0,59] = 1
    truth_one_inside_other[1,149] = 1
    truth_one_inside_other[2,239] = 1
    truth_one_inside_other[3,329] = 1
    one_inside_other = np.hstack(([one_inside_other]*repetitions))
    truth_one_inside_other = np.hstack(([truth_one_inside_other]*repetitions))

    patterns[1] = [one_inside_other, truth_one_inside_other]
    
    # Type 3: Three patterns (but essentially two); one small, one big and one without separation between the two
    
    gap = np.zeros((240,30))
    
    big = np.ones((150,60)) 
    small = np.ones((60,60))
    v1 = np.zeros((90,60))
    v2 = np.zeros((180,60))
    
    
    voc1 = np.vstack((big,v1))
    voc2 = np.vstack((small,v2))

    vocs = [voc1, voc2]
    
    no_separation = np.hstack((voc1, gap, voc2, gap, voc1, voc2, gap))
    truth_no_separation = np.zeros((len(vocs),len(no_separation.T)))
    truth_no_separation[0,59] = 1
    truth_no_separation[1,149] = 1
    truth_no_separation[0,239] = 1
    truth_no_separation[1,299] = 1
    no_separation = np.hstack(([no_separation]*repetitions))
    truth_no_separation = np.hstack(([truth_no_separation]*repetitions))
    
    patterns[2] = [no_separation, truth_no_separation]
    
    
    # Type 4: Two patterns, both with same frequency characteristics but one short and one long in terms of duration
    
    gap = np.zeros((240,30))
    
    big = np.ones((150,60)) 
    v1 = np.zeros((90,60))
    
    
    voc1 = np.vstack((big,v1))

    vocs = [voc1, voc1]
    
    short_and_long = np.hstack((voc1, gap, voc1, voc1, voc1, gap))
    truth_short_and_long = np.zeros((len(vocs),len(short_and_long.T)))
    truth_short_and_long[0,59] = 1
    truth_short_and_long[1,269] = 1
    short_and_long = np.hstack(([short_and_long]*repetitions))
    truth_short_and_long = np.hstack(([truth_short_and_long]*repetitions))
    
    patterns[3] = [short_and_long, truth_short_and_long]
    
    return patterns[choice]