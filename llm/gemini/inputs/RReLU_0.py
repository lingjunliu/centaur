
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def rrelu_inputs():
    list_of_inputs = []

    input = np.random.randn(2).astype(np.float32)
    input_dict = {
        "lower": 0.1,
        "upper": 0.3,
        "inplace": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(2, 3).astype(np.float64)
    input_dict = {
        "lower": 0.0,
        "upper": 1.0,
        "inplace": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "lower": -0.1,
        "upper": 0.2,
        "inplace": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(1, 2, 3, 4).astype(np.float64)
    input_dict = {
        "lower": 0.25,
        "upper": 0.75,
        "inplace": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([-1, 0, 1]).astype(np.float32)
    input_dict = {
        "lower": 0.1,
        "upper": 0.3,
        "inplace": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.RReLU"] = rrelu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.RReLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RReLU'.")

check_valid('torch.nn.RReLU', generated_inputs['torch.nn.RReLU'], lib="torch")
