
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def atleast_2d_inputs():
    list_of_inputs = []

    # Scalar input
    input1 = np.array(5)
    input_dict1 = {"tensors": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # 1D array
    input2 = np.array([1, 2, 3])
    input_dict2 = {"tensors": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # 2D array
    input3 = np.array([[1, 2], [3, 4]])
    input_dict3 = {"tensors": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # 3D array
    input4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict4 = {"tensors": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Float array
    input5 = np.array([1.5, 2.5, 3.5])
    input_dict5 = {"tensors": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Negative values
    input6 = np.array([-1, -2, -3])
    input_dict6 = {"tensors": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Empty array
    input7 = np.array([])
    input_dict7 = {"tensors": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.atleast_2d_1"] = atleast_2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.atleast_2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atleast_2d_1'.")

check_valid('torch.atleast_2d', generated_inputs['torch.atleast_2d_1'], lib="torch")
