
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_where_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean tensor
    condition = np.array([[True, False], [False, True]])
    input_tensor = np.array([[1, 2], [3, 4]])
    other_tensor = np.array([[5, 6], [7, 8]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Boolean tensor with different shape, broadcasting input and other
    condition = np.array([True, False, True])
    input_tensor = np.array([1, 2, 3])
    other_tensor = np.array([0, 0, 0])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D boolean tensor
    condition = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other_tensor = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean tensor with all True values
    condition = np.array([[True, True], [True, True]])
    input_tensor = np.array([[1, 2], [3, 4]])
    other_tensor = np.array([[5, 6], [7, 8]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean tensor with all False values
    condition = np.array([[False, False], [False, False]])
    input_tensor = np.array([[1, 2], [3, 4]])
    other_tensor = np.array([[5, 6], [7, 8]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.where_2"] = torch_where_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.where_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.where_2'.")

check_valid('torch.where', generated_inputs['torch.where_2'], lib="torch")
