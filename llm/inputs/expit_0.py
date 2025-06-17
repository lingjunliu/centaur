
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def expit_inputs():
    list_of_inputs = []

    # Input 1: Simple float tensor
    input1 = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Larger float tensor with more diverse values
    input2 = np.array([-5.0, -2.5, 0.0, 2.5, 5.0, 10.0, -10.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional float tensor
    input3 = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Large multi-dimensional float tensor
    input4 = np.random.randn(3, 4, 5).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Float16 tensor
    input5 = np.array([-1, 0, 1], dtype=np.float16)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.special.expit"] = expit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.expit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.expit'.")

check_valid('torch.special.expit', generated_inputs['torch.special.expit'], lib="torch")
