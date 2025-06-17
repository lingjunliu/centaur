
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cummin_inputs():
    list_of_inputs = []

    # Test case 1: 1D tensor, dim=0
    input1 = torch.randn(10).numpy()
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.cummin"] = cummin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cummin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cummin'.")

check_valid('torch.cummin', generated_inputs['torch.cummin'], lib="torch")
