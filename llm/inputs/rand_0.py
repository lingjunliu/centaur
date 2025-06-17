
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def rand_inputs():
    list_of_inputs = []

    # Input 1: Basic size argument
    input_dict = {"size": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Size as a tuple
    input_dict = {"size": (2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Size as a list
    input_dict = {"size": [4, 2, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With dtype specified
    input_dict = {"size": (3, 4), "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With requires_grad=True
    input_dict = {"size": (2, 2), "requires_grad": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.rand"] = rand_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rand' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rand'.")

check_valid('torch.rand', generated_inputs['torch.rand'], lib="torch")
