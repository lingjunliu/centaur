
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def atleast_2d_inputs():
    list_of_inputs = []

    # Scalar input
    input1 = np.array(5.0)
    list_of_inputs.append({"tensors": [input1]})

    # 1D array
    input2 = np.array([1, 2, 3])
    list_of_inputs.append({"tensors": [input2]})

    # 2D array
    input3 = np.array([[1, 2], [3, 4]])
    list_of_inputs.append({"tensors": [input3]})

    # 3D array
    input4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    list_of_inputs.append({"tensors": [input4]})

    # Multiple inputs: scalar and 1D
    input5 = np.array(2)
    input6 = np.array([4,5,6])
    list_of_inputs.append({"tensors": [input5, input6]})

    # Multiple 2D inputs
    input7 = np.array([[1, 2], [3, 4]])
    input8 = np.array([[5, 6], [7, 8]])
    list_of_inputs.append({"tensors": [input7, input8]})
    
    # Negative values
    input9 = np.array([-1, -2, -3])
    list_of_inputs.append({"tensors": [input9]})
    
    # Float values
    input10 = np.array([1.5, 2.5, 3.5])
    list_of_inputs.append({"tensors": [input10]})

    return list_of_inputs

generated_inputs["torch.atleast_2d_2"] = atleast_2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.atleast_2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atleast_2d_2'.")

check_valid('torch.atleast_2d', generated_inputs['torch.atleast_2d_2'], lib="torch")
