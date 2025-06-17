
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_div_inputs():
    list_of_inputs = []

    # Case 1: Basic float division
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1, "rounding_mode": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer division with rounding_mode='floor'
    input2 = torch.randint(-5, 5, (2, 2)).numpy()
    other2 = torch.randint(1, 5, (2, 2)).numpy()
    input_dict2 = {"input": input2, "other": other2, "rounding_mode": 'floor', "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Integer division with rounding_mode='trunc'
    input3 = torch.randint(-5, 5, (2, 2)).numpy()
    other3 = torch.randint(1, 5, (2, 2)).numpy()
    input_dict3 = {"input": input3, "other": other3, "rounding_mode": 'trunc', "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Broadcasting with a scalar
    input4 = torch.randn(5).numpy()
    other4 = 2.0
    input_dict4 = {"input": input4, "other": other4, "rounding_mode": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Different shapes with broadcasting
    input5 = torch.randn(2, 3, 4).numpy()
    other5 = torch.randn(4).numpy()
    input_dict5 = {"input": input5, "other": other5, "rounding_mode": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.div_1"] = torch_div_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.div_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.div_1'.")

check_valid('torch.div', generated_inputs['torch.div_1'], lib="torch")
