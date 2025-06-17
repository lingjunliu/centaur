
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def asinh_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive and negative floats
    input_1 = np.array([0.5, -1.0, 2.0, -0.75], dtype=np.float32)
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Scalar value
    input_2 = np.array(-2.5, dtype=np.float32)
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Zero value
    input_3 = np.array([0.0], dtype=np.float64)
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D tensor
    input_4 = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.asinh"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.asinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asinh'.")

check_valid('torch.asinh', generated_inputs['torch.asinh'], lib="torch")
