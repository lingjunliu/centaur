
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def searchsorted_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 1D tensors, left side
    sorted_sequence = np.array([1, 3, 5, 7, 9])
    values = np.array([3, 6, 9])
    input_dict = {
        "sorted_sequence": sorted_sequence,
        "values": values,
        "out_int32": False,
        "right": False,
        "side": None,
        "out": None,
        "sorter": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Basic case with 1D tensors, right side
    sorted_sequence = np.array([1, 3, 5, 7, 9])
    values = np.array([3, 6, 9])
    input_dict = {
        "sorted_sequence": sorted_sequence,
        "values": values,
        "out_int32": False,
        "right": True,
        "side": None,
        "out": None,
        "sorter": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D tensors, left side
    sorted_sequence = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    values = np.array([[3, 6, 9], [3, 6, 9]])
    input_dict = {
        "sorted_sequence": sorted_sequence,
        "values": values,
        "out_int32": False,
        "right": False,
        "side": None,
        "out": None,
        "sorter": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D tensors, right side, out_int32=True
    sorted_sequence = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    values = np.array([[3, 6, 9], [3, 6, 9]])
    input_dict = {
        "sorted_sequence": sorted_sequence,
        "values": values,
        "out_int32": True,
        "right": True,
        "side": None,
        "out": None,
        "sorter": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: scalar values
    sorted_sequence = np.array([1, 3, 5, 7, 9])
    values = np.array(6)
    input_dict = {
        "sorted_sequence": sorted_sequence,
        "values": values,
        "out_int32": False,
        "right": False,
        "side": None,
        "out": None,
        "sorter": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.searchsorted"] = searchsorted_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.searchsorted' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.searchsorted'.")

check_valid('torch.searchsorted', generated_inputs['torch.searchsorted'], lib="torch")
