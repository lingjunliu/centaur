
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def Threshold_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3).numpy()
    threshold1 = 0.5
    value1 = 1.0
    inplace1 = False
    input_dict1 = {
        "input": input1,
        "threshold": threshold1,
        "value": value1,
        "inplace": inplace1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 2, 3, 4).numpy()
    threshold2 = -0.2
    value2 = 0.0
    inplace2 = True
    input_dict2 = {
        "input": input2,
        "threshold": threshold2,
        "value": value2,
        "inplace": inplace2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(-5, 5, (5,)).float().numpy()
    threshold3 = 2.0
    value3 = -1.0
    inplace3 = False
    input_dict3 = {
        "input": input3,
        "threshold": threshold3,
        "value": value3,
        "inplace": inplace3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(2, 2, 2).numpy()
    threshold4 = 0.0
    value4 = 2.5
    inplace4 = True
    input_dict4 = {
        "input": input4,
        "threshold": threshold4,
        "value": value4,
        "inplace": inplace4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3,).numpy()
    threshold5 = 1.0
    value5 = -2.0
    inplace5 = False
    input_dict5 = {
        "input": input5,
        "threshold": threshold5,
        "value": value5,
        "inplace": inplace5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1, 1, 1, 1).numpy()
    threshold6 = 0.7
    value6 = 3.0
    inplace6 = True
    input_dict6 = {
        "input": input6,
        "threshold": threshold6,
        "value": value6,
        "inplace": inplace6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(4, 5).numpy()
    threshold7 = -1.5
    value7 = 0.5
    inplace7 = False
    input_dict7 = {
        "input": input7,
        "threshold": threshold7,
        "value": value7,
        "inplace": inplace7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.nn.Threshold"] = Threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Threshold'.")

check_valid('torch.nn.Threshold', generated_inputs['torch.nn.Threshold'], lib="torch")
