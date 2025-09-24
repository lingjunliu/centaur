
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def triplet_margin_loss_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default parameters
    input_dict = {
        "anchor": np.random.randn(100, 128),
        "positive": np.random.randn(100, 128),
        "negative": np.random.randn(100, 128),
        "margin": 1.0,
        "p": 2,
        "eps": 1e-06,
        "swap": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different margin and p
    input_dict = {
        "anchor": np.random.randn(50, 64),
        "positive": np.random.randn(50, 64),
        "negative": np.random.randn(50, 64),
        "margin": 0.5,
        "p": 1,
        "eps": 1e-06,
        "swap": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With swap enabled
    input_dict = {
        "anchor": np.random.randn(20, 32),
        "positive": np.random.randn(20, 32),
        "negative": np.random.randn(20, 32),
        "margin": 1.0,
        "p": 2,
        "eps": 1e-06,
        "swap": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different reduction method
    input_dict = {
        "anchor": np.random.randn(80, 16),
        "positive": np.random.randn(80, 16),
        "negative": np.random.randn(80, 16),
        "margin": 1.0,
        "p": 2,
        "eps": 1e-06,
        "swap": False,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 'none' reduction method
    input_dict = {
        "anchor": np.random.randn(120, 256),
        "positive": np.random.randn(120, 256),
        "negative": np.random.randn(120, 256),
        "margin": 1.0,
        "p": 2,
        "eps": 1e-06,
        "swap": False,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = triplet_margin_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('TripletMarginLoss', generated_inputs)
