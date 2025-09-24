
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ShortStorage_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    input_dict = {"data": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of positive integers
    input_dict = {"data": [1, 2, 3, 4, 5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of negative integers
    input_dict = {"data": [-1, -2, -3, -4, -5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of mixed positive and negative integers
    input_dict = {"data": [-1, 2, -3, 4, -5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of integers with some duplicates
    input_dict = {"data": [1, 2, 3, 2, 1]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Longer list of integers
    input_dict = {"data": list(range(-10, 10))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List with zero
    input_dict = {"data": [-1, 0, 1, 2, -2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.ShortStorage_3"] = ShortStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ShortStorage_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ShortStorage_3'.")

check_valid('torch.ShortStorage', generated_inputs['torch.ShortStorage_3'], lib="torch")
