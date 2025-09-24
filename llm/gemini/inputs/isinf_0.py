
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def isinf_inputs():
    list_of_inputs = []

    input_1 = np.array([1, float('inf'), 2, float('-inf'), float('nan')]).astype(np.float32)
    input_dict = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_2 = np.array([[1, float('inf')], [2, float('-inf')]]).astype(np.float64)
    input_dict = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_3 = np.array([float('inf'), float('-inf'), 0, -0]).astype(np.int32)
    input_dict = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_4 = np.array([1 + 1j * float('inf'), 2 - 1j * float('-inf'), 3 + 0j]).astype(np.complex128)
    input_dict = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_5 = np.array([[[1, float('inf')], [2, float('-inf')]], [[3, 4], [5, 6]]]).astype(np.float32)
    input_dict = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_6 = np.array([float('inf')]).astype(np.float32)
    input_dict = {"input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_7 = np.array([-float('inf')]).astype(np.float64)
    input_dict = {"input": input_7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.isinf"] = isinf_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isinf'.")

check_valid('torch.isinf', generated_inputs['torch.isinf'], lib="torch")
