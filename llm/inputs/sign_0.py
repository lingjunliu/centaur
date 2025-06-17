
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_sign_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with mixed positive, negative, and zero values (float)
    input1 = np.array([0.7, -1.2, 0.0, 2.3, -0.5], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with mixed positive, negative, and zero values (int)
    input2 = np.array([[1, -2, 0], [-3, 4, -5]], dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Scalar value (float)
    input3 = np.array(-3.14, dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D tensor with negative values (float)
    input4 = np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor with only zero values (int)
    input5 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.sign"] = torch_sign_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sign'.")

check_valid('torch.sign', generated_inputs['torch.sign'], lib="torch")
