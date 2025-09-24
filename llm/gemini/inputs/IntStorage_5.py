
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def IntStorage_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    input1 = []
    input_dict1 = {"storage": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: List with positive integers
    input2 = [1, 2, 3, 4, 5]
    input_dict2 = {"storage": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: List with negative integers
    input3 = [-1, -2, -3, -4, -5]
    input_dict3 = {"storage": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: List with mixed positive and negative integers
    input4 = [-1, 2, -3, 4, -5]
    input_dict4 = {"storage": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: List of integers with a different range
    input5 = [100, 200, 300]
    input_dict5 = {"storage": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.IntStorage_5"] = IntStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.IntStorage_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.IntStorage_5'.")

check_valid('torch.IntStorage', generated_inputs['torch.IntStorage_5'], lib="torch")
