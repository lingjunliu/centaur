
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def LogSoftmax_inputs():
    list_of_inputs = []

    input_1 = torch.randn(2, 3).numpy()
    dim_1 = 1
    input_dict_1 = {
        "input": input_1,
        "dim": dim_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = torch.randn(2, 3, 4).numpy()
    dim_2 = 0
    input_dict_2 = {
        "input": input_2,
        "dim": dim_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = torch.randn(1, 5, 2, 2).numpy()
    dim_3 = 2
    input_dict_3 = {
        "input": input_3,
        "dim": dim_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = torch.randn(1, 3).numpy()
    dim_4 = -1
    input_dict_4 = {
        "input": input_4,
        "dim": dim_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = torch.randn(4, 2).numpy()
    dim_5 = 0
    input_dict_5 = {
        "input": input_5,
        "dim": dim_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_6 = torch.randn(2, 2, 2).numpy()
    dim_6 = 1
    input_dict_6 = {
        "input": input_6,
        "dim": dim_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    input_7 = torch.randn(1, 1, 1, 1).numpy()
    dim_7 = 3
    input_dict_7 = {
        "input": input_7,
        "dim": dim_7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_8 = torch.randn(3, 5, 7, 9).numpy()
    dim_8 = -1
    input_dict_8 = {
        "input": input_8,
        "dim": dim_8
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSoftmax"] = LogSoftmax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LogSoftmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LogSoftmax'.")

check_valid('torch.nn.LogSoftmax', generated_inputs['torch.nn.LogSoftmax'], lib="torch")
