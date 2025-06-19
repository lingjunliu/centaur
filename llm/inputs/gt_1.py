
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def gt_inputs():
    list_of_inputs = []

    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other1 = torch.tensor([[1, 1], [4, 4]]).numpy()
    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).numpy()
    other2 = 2.0
    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3, 4).numpy()
    other3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {
        "input": input3,
        "other": other3,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randint(-5, 5, (5,)).numpy()
    other4 = torch.tensor([0]).numpy()
    input_dict4 = {
        "input": input4,
        "other": other4,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([[-1, -2], [-3, -4]], dtype=torch.int64).numpy()
    other5 = -2
    input_dict5 = {
        "input": input5,
        "other": other5,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.gt_1"] = gt_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.gt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gt_1'.")

check_valid('torch.gt', generated_inputs['torch.gt_1'], lib="torch")
