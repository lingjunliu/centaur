
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def nansum_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D tensor with NaN
    input_tensor = np.array([1.0, 2.0, float('nan'), 4.0])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor with NaN
    input_tensor = np.array([[1.0, 2.0], [3.0, float('nan')]])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: All NaN values
    input_tensor = np.array([float('nan'), float('nan'), float('nan')])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: No NaN values
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with NaN
    input_tensor = np.array([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [float('nan'), 8.0]]])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Single element with NaN
    input_tensor = np.array([float('nan')])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single element without NaN
    input_tensor = np.array([42.0])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Negative values with NaN
    input_tensor = np.array([-1.0, -2.0, float('nan'), -4.0])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large 2D tensor with multiple NaNs
    input_tensor = np.array([[1.0, 2.0, 3.0], [float('nan'), 5.0, float('nan')], [7.0, 8.0, 9.0]])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Mix of positive, negative, zero, and NaN
    input_tensor = np.array([10.0, -5.0, 0.0, float('nan'), 3.14, -2.71])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_1"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_1'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_1'], lib="torch", suffix=1)
