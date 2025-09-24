
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def round_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array
    input = np.array([1.2, 2.7, 3.5, 4.1, 5.9], dtype=np.float32)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array
    input = np.array([[1.2, 2.7], [3.5, 4.1]], dtype=np.float32)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Array with negative values
    input = np.array([-1.2, -2.7, 3.5, -4.1, 5.9], dtype=np.float32)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Array with zero
    input = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with large values
    input = np.array([100.2, 200.7, -300.5], dtype=np.float32)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with small values
    input = np.array([0.0001, 0.0009, -0.0005], dtype=np.float32)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array
    input = np.array([[[1.2, 2.7], [3.5, 4.1]], [[5.9, 6.2], [7.7, 8.5]]], dtype=np.float32)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with mixed positive and negative values including 0
    input = np.array([-2.3, 0, 1.8, -0.5, 3.1], dtype=np.float32)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Float64 array
    input = np.array([1.2, 2.7, 3.5, 4.1, 5.9], dtype=np.float64)
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.round"] = round_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.round' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.round'.")

check_valid('torch.special.round', generated_inputs['torch.special.round'], lib="torch", suffix=0)
