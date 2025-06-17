
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def log10_inputs():
    list_of_inputs = []

    # Input 1: Basic positive float tensor
    input1 = torch.rand(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.log10"] = log10_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.log10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log10'.")

check_valid('torch.log10', generated_inputs['torch.log10'], lib="torch")
