
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def amax_inputs():
    list_of_inputs = []

    # Input 1: Basic case with dim=1 and keepdim=False
    input1 = torch.randn(4, 4).numpy()
    dim1 = (1,)
    keepdim1 = False
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.amax_2"] = amax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.amax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amax_2'.")

check_valid('torch.amax', generated_inputs['torch.amax_2'], lib="torch")
