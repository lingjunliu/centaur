
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def logical_not_inputs():
    list_of_inputs = []

    # Input 1: Boolean array
    input1 = np.array([True, False])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer array (0 and 1 for logical interpretation)
    input2 = np.array([0, 1], dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multidimensional boolean array
    input3 = np.array([[True, False], [False, True]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D boolean array
    input4 = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float array (0.0 and 1.0 for logical interpretation)
    input5 = np.array([0.0, 1.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.logical_not"] = logical_not_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logical_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_not'.")

check_valid('torch.logical_not', generated_inputs['torch.logical_not'], lib="torch")
