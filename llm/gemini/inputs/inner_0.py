
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def inner_inputs():
    list_of_inputs = []

    # Case 1: 1D tensors
    input1 = np.array([1, 2, 3])
    input2 = np.array([0, 2, 1])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Multidimensional tensors
    input1 = np.random.randn(2, 3)
    input2 = np.random.randn(2, 3)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Scalar input
    input1 = np.random.randn(2, 3)
    input2 = np.array(2.0)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Integer tensors
    input1 = np.array([1, 2, 3], dtype=np.int64)
    input2 = np.array([0, 2, 1], dtype=np.int64)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values
    input1 = np.array([-1, 2, -3])
    input2 = np.array([0, -2, 1])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.inner"] = inner_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.inner' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.inner'.")

check_valid('torch.inner', generated_inputs['torch.inner'], lib="torch")
