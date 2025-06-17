
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def range_inputs():
    list_of_inputs = []

    input_dict = {
        "start": 1.0,
        "end": 4.0,
        "step": 1.0,
        "out": None,
        "dtype": None,
        "layout": "strided",
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.range"] = range_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.range' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.range'.")

check_valid('torch.range', generated_inputs['torch.range'], lib="torch")
