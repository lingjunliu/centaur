
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def rfft_inputs():
    list_of_inputs = []

    input1 = torch.arange(4).numpy()
    input_dict1 = {
        "input": input1,
        "n": None,
        "dim": -1,
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(5).numpy()
    input_dict2 = {
        "input": input2,
        "n": 10,
        "dim": 0,
        "norm": "forward",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 4).numpy()
    input_dict3 = {
        "input": input3,
        "n": None,
        "dim": 1,
        "norm": "backward",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {
        "input": input4,
        "n": 2,
        "dim": 2,
        "norm": "ortho",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2, 2, 2).numpy()
    input_dict5 = {
        "input": input5,
        "n": None,
        "dim": 3,
        "norm": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.fft.rfft"] = rfft_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fft.rfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.rfft'.")

check_valid('torch.fft.rfft', generated_inputs['torch.fft.rfft'], lib="torch")
