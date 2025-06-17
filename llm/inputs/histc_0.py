
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def histc_inputs():
    list_of_inputs = []

    input1 = np.array([1., 2, 1])
    bins1 = 4
    min1 = 0.
    max1 = 3.
    input_dict1 = {
        "input": input1,
        "bins": bins1,
        "min": min1,
        "max": max1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(10)
    bins2 = 5
    min2 = -1.
    max2 = 1.
    input_dict2 = {
        "input": input2,
        "bins": bins2,
        "min": min2,
        "max": max2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randint(0, 10, (20,)).astype(float)
    bins3 = 10
    min3 = 2.
    max3 = 8.
    input_dict3 = {
        "input": input3,
        "bins": bins3,
        "min": min3,
        "max": max3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(5, 5)
    bins4 = 20
    min4 = -2.
    max4 = 2.
    input_dict4 = {
        "input": input4,
        "bins": bins4,
        "min": min4,
        "max": max4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1, 2, 3, 4, 5], dtype=np.int32).astype(float)
    bins5 = 6
    min5 = 0.
    max5 = 6.
    input_dict5 = {
        "input": input5,
        "bins": bins5,
        "min": min5,
        "max": max5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.histc"] = histc_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.histc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histc'.")

check_valid('torch.histc', generated_inputs['torch.histc'], lib="torch")
