
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ConstantPad1d_inputs():
    list_of_inputs = []

    # Test case 1: Single integer padding
    input1 = torch.randn(1, 2, 4).numpy()
    padding1 = 2
    value1 = 3.5
    input_dict1 = {"padding": padding1, "value": value1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Tuple padding (different left and right)
    input2 = torch.randn(1, 2, 3).numpy()
    padding2 = (3, 1)
    value2 = -1.0
    input_dict2 = {"padding": padding2, "value": value2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Integer input tensor
    input3 = torch.randint(0, 10, (1, 1, 5)).float().numpy()
    padding3 = 1
    value3 = 0.0
    input_dict3 = {"padding": padding3, "value": value3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Different input dimensions (C, Win)
    input4 = torch.randn(2, 5).numpy()
    padding4 = 2
    value4 = 1.2
    input_dict4 = {"padding": padding4, "value": value4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Negative padding value
    input5 = torch.randn(1, 3, 2).numpy()
    padding5 = 1
    value5 = -2.5
    input_dict5 = {"padding": padding5, "value": value5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ConstantPad1d_1"] = ConstantPad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConstantPad1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad1d_1'.")

check_valid('torch.nn.ConstantPad1d', generated_inputs['torch.nn.ConstantPad1d_1'], lib="torch")
