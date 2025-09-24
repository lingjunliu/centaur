
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_median_inputs():
    list_of_inputs = []

    # Case 1: 2D tensor, dim=0, keepdim=False
    input_2 = torch.randn(4, 5).numpy()
    input_dict_2 = {
        "input": input_2,
        "dim": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    
    return list_of_inputs

generated_inputs["torch.median_2"] = torch_median_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.median_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.median_2'.")

check_valid('torch.median', generated_inputs['torch.median_2'], lib="torch")
