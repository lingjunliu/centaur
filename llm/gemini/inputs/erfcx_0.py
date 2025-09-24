
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def erfcx_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative values
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Larger values
    input3 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multidimensional tensor
    input4 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with very small values
    input5 = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.special.erfcx"] = erfcx_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.erfcx' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.erfcx'.")

check_valid('torch.special.erfcx', generated_inputs['torch.special.erfcx'], lib="torch")
