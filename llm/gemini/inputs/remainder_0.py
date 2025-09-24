
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def remainder_inputs():
    list_of_inputs = []

    input1 = torch.tensor([-3., -2, -1, 1, 2, 3]).numpy()
    other1 = 2
    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other2 = -1.5
    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3).numpy()
    other3 = torch.randn(2, 3).numpy()
    input_dict3 = {
        "input": input3,
        "other": other3,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randint(-10, 10, (5,)).float().numpy()
    other4 = 3.0
    input_dict4 = {
        "input": input4,
        "other": other4,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randint(-10, 10, (2, 2, 2)).numpy()
    other5 = torch.tensor([-2, -3]).numpy()
    input_dict5 = {
        "input": input5,
        "other": other5,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]).numpy()
    other6 = torch.tensor([2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7]).numpy()
    input_dict6 = {
        "input": input6,
        "other": other6,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(3, 4, dtype=torch.float64).numpy()
    other7 = 2.5
    input_dict7 = {
        "input": input7,
        "other": other7,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.remainder"] = remainder_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.remainder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.remainder'.")

check_valid('torch.remainder', generated_inputs['torch.remainder'], lib="torch")
