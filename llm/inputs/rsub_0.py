
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def rsub_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    alpha1 = 1.0
    input_dict1 = {"input": input1, "other": other1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors with alpha
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
    alpha2 = 2.0
    input_dict2 = {"input": input2, "other": other2, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Negative values and different dimensions
    input3 = np.array([-1.0, -2.0, -3.0])
    other3 = 5.0
    alpha3 = 0.5
    input_dict3 = {"input": input3, "other": other3, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Scalar other
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other4 = 5.0
    alpha4 = 1.0
    input_dict4 = {"input": input4, "other": other4, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 3D tensor
    input5 = np.random.rand(2, 3, 4)
    other5 = 2.0
    alpha5 = 1.0
    input_dict5 = {"input": input5, "other": other5, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.rsub"] = rsub_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rsub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rsub'.")

check_valid('torch.rsub', generated_inputs['torch.rsub'], lib="torch")
