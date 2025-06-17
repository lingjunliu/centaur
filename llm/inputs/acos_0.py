
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def acos_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor with values in [-1, 1]
    input1 = np.array([0.5], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.acos"] = acos_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.acos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.acos'.")

check_valid('torch.acos', generated_inputs['torch.acos'], lib="torch")
