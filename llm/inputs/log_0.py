
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_log_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with positive floats
    input_1 = np.array([1.0, 2.718, 7.389, 20.086, 54.598], dtype=np.float32)
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor with positive floats
    input_2 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Scalar tensor
    input_3 = np.array(5.0, dtype=np.float32)
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: 3D tensor
    input_4 = np.random.rand(2, 3, 4).astype(np.float32) + 0.1
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: Large values
    input_5 = np.array([1e5, 1e10, 1e15], dtype=np.float64)
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Very small values, greater than zero
    input_6 = np.array([1e-5, 1e-10, 1e-15], dtype=np.float64)
    input_dict_6 = {"input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.log"] = torch_log_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log'.")

check_valid('torch.log', generated_inputs['torch.log'], lib="torch")
