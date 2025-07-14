
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def where_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D boolean tensor
    condition = np.array([True, False, True, False], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean tensor
    condition = np.array([[True, False], [False, True]], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean tensor
    condition = np.array([[[True, False], [False, True]], [[False, True], [True, False]]], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty boolean tensor
    condition = np.array([], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean tensor with all True values
    condition = np.array([True, True, True], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean tensor with all False values
    condition = np.array([False, False, False], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean tensor with mixed True/False in a larger array
    condition = np.array([[True, False, True], [False, True, False], [True, True, False]], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D boolean tensor with more elements
    condition = np.array([True, False, True, False, True, True, False, False], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boolean tensor with a different shape (1, N)
    condition = np.array([[True, False, True, True]], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean tensor with a different shape (N, 1)
    condition = np.array([[True], [False], [True], [False]], dtype=bool)
    input_dict = {"condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.where_2"] = where_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.where_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.where_2'.")

check_valid('torch.where', generated_inputs['torch.where_2'], lib="torch", suffix=2)
