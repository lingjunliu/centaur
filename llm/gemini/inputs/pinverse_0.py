
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def pinverse_inputs():
    list_of_inputs = []

    # Input 1: Simple square matrix
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict1 = {"input": input1, "rcond": 1e-15}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Rectangular matrix (more rows than columns)
    input2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    input_dict2 = {"input": input2, "rcond": 1e-15}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Rectangular matrix (more columns than rows)
    input3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_dict3 = {"input": input3, "rcond": 1e-15}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Matrix with some near-zero singular values
    input4 = np.array([[1.0, 1.0], [1.0, 1.0]])
    input_dict4 = {"input": input4, "rcond": 1e-3}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D Tensor
    input5 = np.random.rand(2, 3, 4)
    input_dict5 = {"input": input5, "rcond": 1e-15}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.pinverse"] = pinverse_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.pinverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pinverse'.")

check_valid('torch.pinverse', generated_inputs['torch.pinverse'], lib="torch")
