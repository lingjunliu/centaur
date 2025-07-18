
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def normal_4_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case, 1D output
    size_1 = (5,)
    input_dict = {
        'mean': 0.0,
        'std': 1.0,
        'size': size_1,
        'out': np.zeros(size_1, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D output
    size_2 = (2, 3)
    input_dict = {
        'mean': 10.0,
        'std': 2.0,
        'size': size_2,
        'out': np.zeros(size_2, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Negative mean
    size_3 = (4,)
    input_dict = {
        'mean': -5.5,
        'std': 0.5,
        'size': size_3,
        'out': np.zeros(size_3, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D output
    size_4 = (2, 2, 2)
    input_dict = {
        'mean': 1.2,
        'std': 3.4,
        'size': size_4,
        'out': np.zeros(size_4, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Zero standard deviation
    size_5 = (3, 2)
    input_dict = {
        'mean': 7.0,
        'std': 0.0,
        'size': size_5,
        'out': np.zeros(size_5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large values for mean and std
    size_6 = (1, 6)
    input_dict = {
        'mean': 1000.0,
        'std': 500.0,
        'size': size_6,
        'out': np.zeros(size_6, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Small values for mean and std
    size_7 = (4, 1)
    input_dict = {
        'mean': 1e-5,
        'std': 1e-6,
        'size': size_7,
        'out': np.zeros(size_7, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Scalar output (empty size tuple)
    size_8 = ()
    input_dict = {
        'mean': -1.0,
        'std': 1.0,
        'size': size_8,
        'out': np.zeros(size_8, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another scalar output
    size_9 = ()
    input_dict = {
        'mean': 3.14,
        'std': 2.71,
        'size': size_9,
        'out': np.zeros(size_9, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 3D tensor
    size_10 = (3, 4, 5)
    input_dict = {
        'mean': 0.0,
        'std': 1.0,
        'size': size_10,
        'out': np.zeros(size_10, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.normal_4"] = normal_4_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.normal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_4'.")

check_valid('torch.normal', generated_inputs['torch.normal_4'], lib="torch", suffix=4)
