
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def pairwise_distance_inputs():
    list_of_inputs = []

    # Case 1: Basic case with N, D dimensions, p=2, default eps, keepdim=False
    input1 = np.random.randn(10, 5).astype(np.float32)
    input2 = np.random.randn(10, 5).astype(np.float32)
    input_dict = {
        "p": 2.0,
        "eps": 1e-06,
        "keepdim": False,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Single dimension D, p=1, keepdim=True
    input1 = np.random.randn(5).astype(np.float32)
    input2 = np.random.randn(5).astype(np.float32)
    input_dict = {
        "p": 1.0,
        "eps": 1e-06,
        "keepdim": True,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: N, D dimensions with negative p, custom eps
    input1 = np.random.randn(5, 3).astype(np.float32)
    input2 = np.random.randn(5, 3).astype(np.float32)
    input_dict = {
        "p": -1.5,
        "eps": 1e-04,
        "keepdim": False,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: N, D dimensions with p=0, keepdim=True
    input1 = np.random.randn(3, 4).astype(np.float32)
    input2 = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "p": 0.0,
        "eps": 1e-06,
        "keepdim": True,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different shapes for inputs, but compatible
    input1 = np.random.randn(8, 7).astype(np.float32)
    input2 = np.random.randn(8, 7).astype(np.float32)
    input_dict = {
        "p": 3.0,
        "eps": 1e-08,
        "keepdim": False,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.PairwiseDistance"] = pairwise_distance_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.PairwiseDistance', generated_inputs['torch.nn.PairwiseDistance'], lib="torch")
