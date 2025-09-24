
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy

def torch_device_inputs():
    list_of_inputs = []

    input_dict = {
        "type": "cpu",
        "index": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "type": "cuda",
        "index": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "type": "cuda",
        "index": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "type": "mps",
        "index": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "type": "cpu",
        "index": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.device_1"] = torch_device_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.device_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.device_1'.")

check_valid('torch.device', generated_inputs['torch.device_1'], lib="torch")
