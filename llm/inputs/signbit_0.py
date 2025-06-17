
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def signbit_inputs():
    list_of_inputs = []

    # Example 1: Float tensor with positive and negative values
    a = torch.tensor([0.7, -1.2, 0., 2.3]).numpy()
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Float tensor with signed zeros
    a = torch.tensor([-0.0, 0.0]).numpy()
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.signbit"] = signbit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.signbit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.signbit'.")

check_valid('torch.signbit', generated_inputs['torch.signbit'], lib="torch")
