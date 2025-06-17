
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def erfinv_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([0.0, 0.5, 0.9]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor with negative values
    input2 = np.array([-0.5, -0.2, 0.2]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multidimensional float tensor
    input3 = np.array([[0.1, 0.2], [0.3, 0.4]]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar float
    input4 = np.array(0.5).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    return list_of_inputs

generated_inputs["torch.special.erfinv"] = erfinv_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.erfinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.erfinv'.")

check_valid('torch.special.erfinv', generated_inputs['torch.special.erfinv'], lib="torch")
