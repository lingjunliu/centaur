
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cummax_inputs():
    list_of_inputs = []

    # Test case 1: 1D tensor, dim=0
    input_tensor = torch.randn(10).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 2D tensor, dim=0
    input_tensor = torch.randn(5, 5).numpy()
    dim = 0
    input_dict = {"input": input_tensor, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cummax"] = cummax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cummax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cummax'.")

check_valid('torch.cummax', generated_inputs['torch.cummax'], lib="torch")
