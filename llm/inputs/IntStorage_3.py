
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def IntStorage_inputs():
    list_of_inputs = []

    # 1. Empty list
    input1 = {"source": []}
    list_of_inputs.append(copy.deepcopy(input1))

    # 2. List of positive integers
    input2 = {"source": [1, 2, 3, 4, 5]}
    list_of_inputs.append(copy.deepcopy(input2))

    # 3. List of negative integers
    input3 = {"source": [-1, -2, -3, -4, -5]}
    list_of_inputs.append(copy.deepcopy(input3))

    # 4. List of mixed positive and negative integers
    input4 = {"source": [-1, 2, -3, 4, -5]}
    list_of_inputs.append(copy.deepcopy(input4))

    # 5. List of integers with duplicates
    input5 = {"source": [1, 2, 2, 3, 3, 3]}
    list_of_inputs.append(copy.deepcopy(input5))

    return list_of_inputs

generated_inputs["torch.IntStorage_3"] = IntStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.IntStorage_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.IntStorage_3'.")

check_valid('torch.IntStorage', generated_inputs['torch.IntStorage_3'], lib="torch")
