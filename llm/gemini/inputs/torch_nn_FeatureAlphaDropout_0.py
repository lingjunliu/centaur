
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def feature_alpha_dropout_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.randn(20, 16, 4, 32, 32).astype(np.float32)
    p = 0.2
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.randn(10, 8, 1, 16, 16).astype(np.float32)
    p = 0.8
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randn(5, 4, 2, 8, 8).astype(np.float32)
    p = 0.5
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_tensor = np.random.randn(1, 2, 1, 4, 4).astype(np.float32)
    p = 0.1
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.randn(32, 64, 16, 64, 64).astype(np.float32)
    p = 0.9
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (2D input)
    input_tensor = np.random.randn(16, 32).astype(np.float32)
    p = 0.3
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (3D input)
    input_tensor = np.random.randn(8, 16, 32).astype(np.float32)
    p = 0.7
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (4D input)
    input_tensor = np.random.randn(4, 8, 16, 32).astype(np.float32)
    p = 0.4
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (C, D, H, W)
    input_tensor = np.random.randn(3, 4, 5, 6).astype(np.float32)
    p = 0.6
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (N, C, D, H, W) with low p
    input_tensor = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    p = 0.01
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 (N, C, D, H, W) with high p
    input_tensor = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    p = 0.99
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.FeatureAlphaDropout"] = feature_alpha_dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.FeatureAlphaDropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.FeatureAlphaDropout'.")

check_valid('torch.nn.FeatureAlphaDropout', generated_inputs['torch.nn.FeatureAlphaDropout'], lib="torch", suffix=0)
