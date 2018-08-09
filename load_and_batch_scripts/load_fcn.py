#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:04:36 2018

@author: dawnstear
"""

import numpy as np
import os
from inspect import signature
import datetime
import subprocess # replaces os.system()


def load(filepath/data,Transpose=True,normalize=1,compressed=True,sizecheck=True,cluster=False/localclusterfilepath): # setcluster=Nones
    """  Load Single Cell RNA seq gene expression matrix 
    
    Parameters
    ----------
    filepath : str
            A path to the directory where your expression matrix is located
    cluster : str        
            A path to the local cluster where the data matrix will be uploaded to
            
            returns numpy array
    """
    """
    sig = signature(load) # only works after youve called it,, use to test
    subprocess.run(["ls", "-l", "/dev/null"], capture_output=True) # os.system() deprecated
    if cluster:
        subprocess.run(["Upload filepath/data to localclusterfilepath"],capture_output=True,check=True)
                
    for arg in sys.argv[1:]:
    try:
        f = open(arg, 'r')
    except OSError:
        print('cannot open', arg)
    else:
        print(arg, 'has', len(f.readlines()), 'lines')
        f.close()
    """
    
    
    # CHECK ACTUAL INPUT PARAMETER(S)
    try: 
        filepath  # check for existence,   do we need break ?
        break
    except Exception as e:
            raise ValueError("Must enter a filepath")  # valueError type ? 
    try: 
        isinstance(filepath, str)
        break
    # check all different format types here    
    except Exception as e:
            raise ValueError("Arguement 'filepath' must be a string")
            
            
            
    # LOAD DATA AND CHECK THINGS    
    with np.load(filepath) as data: # use np.load? or pd.read_* ?
        sparsity = 0
        # check for successful load
            
            #check that there are only numeric entrys BESIDES, iterate through every entry?
            try:
                for i in range(np.size(data,axis=0)):
                    for j in range(np.size(data,axis=1)):
                        if != data[i,j] # != 0 0.0,,, 0.0==False ?
                        sparsity+=1
                        a = int(data[i+1,j+1]) # +1 so we skip row & col names
                break
            except Exception as e:
                raise ValueError("Data matrix contains")
            # check sparsity and convert to sparse matrix if needed
            if sparsity>0.5*np.size(data):
                #convert to sparse matrix
            
            
            data_nda = np.ndarray(data)  # need pandas ???
            return data_nda
         
            
            