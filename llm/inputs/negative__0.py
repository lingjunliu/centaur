
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def negative__inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor
    input2 = np.array([[1, -2], [3, -4]], dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D long tensor
    input3 = np.array([[[1], [-2]], [[3], [-4]]], dtype=np.int64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar float
    input4 = np.array(-5.5, dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D complex tensor
    input5 = np.array([1 + 1j, -2 - 2j, 3 + 0j], dtype=np.complex64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.negative_"] = negative__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.negative_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.negative_'.")

check_valid('torch.negative_', generated_inputs['torch.negative_'], lib="torch")
