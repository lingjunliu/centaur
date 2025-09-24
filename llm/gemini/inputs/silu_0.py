
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def silu_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, positive values
    input1 = np.random.rand(3, 4).astype(np.float32)
    input_dict1 = {"input": input1, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor, negative values
    input2 = np.random.randn(2, 5).astype(np.float64)
    input_dict2 = {"input": input2, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 4: Multi-dimensional tensor
    input4 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict4 = {"input": input4, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Scalar tensor
    input5 = np.array(-2.5, dtype=np.float64)
    input_dict5 = {"input": input5, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Zero tensor
    input6 = np.zeros((2, 2), dtype=np.float32)
    input_dict6 = {"input": input6, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Float16 tensor
    input7 = np.random.randn(3, 4).astype(np.float16)
    input_dict7 = {"input": input7, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.silu"] = silu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.silu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.silu'.")

check_valid('torch.nn.functional.silu', generated_inputs['torch.nn.functional.silu'], lib="torch")
