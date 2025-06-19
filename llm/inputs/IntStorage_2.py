
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def IntStorage_inputs():
    list_of_inputs = []

    input1 = (0,)
    input_dict1 = {
        "size": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = (1,)
    input_dict2 = {
        "size": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = (5,)
    input_dict3 = {
        "size": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = (10,)
    input_dict4 = {
        "size": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = (100,)
    input_dict5 = {
        "size": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = (2,3)
    input_dict6 = {
        "size": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = (2,3,4)
    input_dict7 = {
        "size": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.IntStorage_2"] = IntStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.IntStorage_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.IntStorage_2'.")

check_valid('torch.IntStorage', generated_inputs['torch.IntStorage_2'], lib="torch")
