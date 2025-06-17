
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def get_default_device_inputs():
    list_of_inputs = []

    input1_dict = {}
    list_of_inputs.append(copy.deepcopy(input1_dict))

    input2_dict = {}
    list_of_inputs.append(copy.deepcopy(input2_dict))

    input3_dict = {}
    list_of_inputs.append(copy.deepcopy(input3_dict))

    input4_dict = {}
    list_of_inputs.append(copy.deepcopy(input4_dict))

    input5_dict = {}
    list_of_inputs.append(copy.deepcopy(input5_dict))
    
    return list_of_inputs

generated_inputs["torch.get_default_device"] = get_default_device_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.get_default_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_default_device'.")

check_valid('torch.get_default_device', generated_inputs['torch.get_default_device'], lib="torch")
