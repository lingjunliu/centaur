
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_autocast_inputs():
    list_of_inputs = []

    input_dict = {
        "device_type": "cuda",
        "enabled": True,
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["torch.autocast_1"] = torch_autocast_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.autocast_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.autocast_1'.")

check_valid('torch.autocast', generated_inputs['torch.autocast_1'], lib="torch")
