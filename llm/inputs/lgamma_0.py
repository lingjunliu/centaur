
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def lgamma_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive floats
    input1 = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with positive floats
    input2 = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor with positive and negative floats. Negative values are valid as torch.lgamma computes ln(abs(gamma(x)))
    input3 = np.array([-2.5, -1.0, 0.5, 1.0, 2.5], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar tensor
    input4 = np.array(3.0, dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor
    input5 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.lgamma"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lgamma'.")

check_valid('torch.lgamma', generated_inputs['torch.lgamma'], lib="torch")
