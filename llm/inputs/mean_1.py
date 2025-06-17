
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor
    input2 = torch.randint(-5, 5, (2, 2), dtype=torch.int32).numpy()
    input_dict2 = {"input": input2, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor
    input3 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Empty tensor (should return NaN)
    input4 = torch.empty(0).numpy()
    input_dict4 = {"input": input4, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor with specific dtype
    input5 = torch.randn(2, 2, 2).numpy()
    input_dict5 = {"input": input5, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Tensor with negative values
    input6 = torch.randn(5).numpy() * -1
    input_dict6 = {"input": input6, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger tensor
    input7 = torch.randn(10, 10).numpy()
    input_dict7 = {"input": input7, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.mean_1"] = torch_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.mean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mean_1'.")

check_valid('torch.mean', generated_inputs['torch.mean_1'], lib="torch")
