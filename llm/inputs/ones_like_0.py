
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ones_like_inputs():
    list_of_inputs = []

    # Input 1: Float tensor
    input1 = np.random.randn(2, 3).astype(np.float32)
    input_dict1 = {"input": input1, "dtype": None, "layout": None, "requires_grad": False, "memory_format": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor
    input2 = np.random.randint(0, 10, size=(3, 4), dtype=np.int64)
    input_dict2 = {"input": input2, "dtype": None, "layout": None, "requires_grad": False, "memory_format": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Bool tensor
    input3 = np.random.choice([True, False], size=(2, 2), p=[0.5, 0.5])
    input_dict3 = {"input": input3, "dtype": None, "layout": None, "requires_grad": False, "memory_format": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Complex tensor
    input4 = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    input_dict4 = {"input": input4, "dtype": None, "layout": None, "requires_grad": False, "memory_format": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor
    input5 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict5 = {"input": input5, "dtype": None, "layout": None, "requires_grad": False, "memory_format": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Tensor with negative values
    input6 = np.random.randint(-10, 0, size=(2, 2), dtype=np.int32)
    input_dict6 = {"input": input6, "dtype": None, "layout": None, "requires_grad": False, "memory_format": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 0-dimensional tensor (scalar)
    input7 = np.array(5.0).astype(np.float64)
    input_dict7 = {"input": input7, "dtype": None, "layout": None, "requires_grad": False, "memory_format": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.ones_like"] = ones_like_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ones_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ones_like'.")

check_valid('torch.ones_like', generated_inputs['torch.ones_like'], lib="torch")
