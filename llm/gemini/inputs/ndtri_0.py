
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ndtri_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([0.1, 0.5, 0.9]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Larger float tensor with different values
    input2 = np.array([[0.01, 0.25, 0.75], [0.2, 0.5, 0.8]]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Tensor with values near 0 and 1, and some intermediate values.
    input3 = np.array([1e-6, 0.5, 0.999999]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Multidimensional array
    input4 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Single value
    input5 = np.array(0.6).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.special.ndtri"] = ndtri_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.ndtri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.ndtri'.")

check_valid('torch.special.ndtri', generated_inputs['torch.special.ndtri'], lib="torch")
