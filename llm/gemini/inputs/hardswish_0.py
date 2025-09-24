
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def hardswish_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor with negative values, converted to float
    input2 = torch.randint(-5, 5, (2, 2)).float().numpy()
    input_dict2 = {"input": input2, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Scalar input
    input3 = np.array(-2.5)
    input_dict3 = {"input": input3, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D float tensor with a mix of positive and negative values
    input4 = torch.randn(2, 3, 5) * 5 - 2.5
    input4 = input4.numpy()
    input_dict4 = {"input": input4, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large tensor
    input5 = torch.randn(10, 10, 10, 10).numpy()
    input_dict5 = {"input": input5, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.functional.hardswish"] = hardswish_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.hardswish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardswish'.")

check_valid('torch.nn.functional.hardswish', generated_inputs['torch.nn.functional.hardswish'], lib="torch")
