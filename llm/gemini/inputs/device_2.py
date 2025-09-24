
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_device_inputs():
    list_of_inputs = []

    input1 = "cpu"
    input_dict1 = {
        "obj": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = "cuda" if torch.cuda.is_available() else "cpu"
    input_dict2 = {
        "obj": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = "cuda:0" if torch.cuda.is_available() else "cpu"
    input_dict3 = {
        "obj": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = "mps" if torch.backends.mps.is_available() else "cpu"
    input_dict4 = {
        "obj": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.device_2"] = torch_device_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.device_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.device_2'.")

check_valid('torch.device', generated_inputs['torch.device_2'], lib="torch")
