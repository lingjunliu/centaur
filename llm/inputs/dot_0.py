
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_dot_inputs():
    list_of_inputs = []

    # Test case 1: Basic integer tensors
    input1 = np.array([2, 3], dtype=np.int64)
    input2 = np.array([2, 1], dtype=np.int64)
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dot"] = torch_dot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dot'.")

check_valid('torch.dot', generated_inputs['torch.dot'], lib="torch")
