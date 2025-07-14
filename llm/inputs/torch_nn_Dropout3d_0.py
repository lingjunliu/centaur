
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dropout3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    p_val = 0.2
    inplace_val = False
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.randn(1, 1, 2, 2, 2).astype(np.float32)
    p_val = 0.5
    inplace_val = True
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randn(5, 4, 3, 2, 1).astype(np.float32)
    p_val = 0.0
    inplace_val = False
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_tensor = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
    p_val = 1.0
    inplace_val = True
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.randn(3, 3, 3, 3, 3).astype(np.float32)
    p_val = 0.8
    inplace_val = False
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.randn(1, 1, 1, 1, 1).astype(np.float32)
    p_val = 0.3
    inplace_val = True
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = np.random.randn(4, 5, 2, 7, 3).astype(np.float32)
    p_val = 0.7
    inplace_val = False
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.randn(2, 1, 5, 3, 4).astype(np.float32)
    p_val = 0.1
    inplace_val = True
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = np.random.randn(1, 3, 1, 3, 1).astype(np.float32)
    p_val = 0.6
    inplace_val = False
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.randn(8, 2, 6, 4, 2).astype(np.float32)
    p_val = 0.9
    inplace_val = True
    input_dict = {"p": p_val, "inplace": inplace_val, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Dropout3d"] = dropout3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Dropout3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Dropout3d'.")

check_valid('torch.nn.Dropout3d', generated_inputs['torch.nn.Dropout3d'], lib="torch", suffix=0)
