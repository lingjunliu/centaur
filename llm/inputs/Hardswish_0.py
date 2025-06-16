
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def Hardswish_inputs():
    list_of_inputs = []

    input1 = np.random.randn(2).astype(np.float32)
    input_dict1 = {
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 3).astype(np.float64)
    input_dict2 = {
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(2, 3, 4).astype(np.float16)
    input_dict3 = {
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.randn(1, 1, 1, 1).astype(np.float32)
    input_dict4 = {
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-4, -2, 0, 2, 4]).astype(np.float32)
    input_dict5 = {
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.randn(5).astype(np.float32)
    input_dict6 = {
        "inplace": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.randn(1, 5, 5, 5).astype(np.float32)
    input_dict7 = {
        "inplace": False,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Hardswish"] = Hardswish_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.Hardswish', generated_inputs['torch.nn.Hardswish'], lib="torch")
