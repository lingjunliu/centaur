
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_log_inputs():
    list_of_inputs = []

    # Test case 1: Basic 1D tensor
    input1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 2D tensor
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))


    return list_of_inputs

generated_inputs["torch.log"] = torch_log_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log'.")

check_valid('torch.log', generated_inputs['torch.log'], lib="torch")
