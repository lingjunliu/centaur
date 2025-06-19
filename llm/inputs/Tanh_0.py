
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def Tanh_inputs():
    list_of_inputs = []

    # Input 1: Scalar
    input_dict = {"input": np.array(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array
    input_dict = {"input": np.array([-1.0, 0.0, 1.0, 2.0])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array
    input_dict = {"input": np.array([[-1.0, 0.0], [1.0, 2.0]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array
    input_dict = {"input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with negative values
    input_dict = {"input": np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 array
    input_dict = {"input": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int32 array
    input_dict = {"input": np.array([-1, 0, 1], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Tanh"] = Tanh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Tanh'.")

check_valid('torch.nn.Tanh', generated_inputs['torch.nn.Tanh'], lib="torch")
