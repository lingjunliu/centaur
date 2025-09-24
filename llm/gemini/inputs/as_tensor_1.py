
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def as_tensor_inputs():
    list_of_inputs = []

    # Input 1: 1D numpy array of integers
    data = np.array([1, 2, 3, 4, 5])
    input_dict = {"data": data, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D numpy array of floats
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"data": data, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D numpy array of complex numbers
    data = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]])
    input_dict = {"data": data, "dtype": torch.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numpy array with negative values
    data = np.array([-1, -2, 0, 1, 2])
    input_dict = {"data": data, "dtype": torch.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty numpy array
    data = np.array([])
    input_dict = {"data": data, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Multi-dimensional array
    data = np.random.rand(2, 3, 4, 5)
    input_dict = {"data": data, "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean numpy array
    data = np.array([True, False, True, True, False])
    input_dict = {"data": data, "dtype": torch.bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.as_tensor_1"] = as_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.as_tensor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_tensor_1'.")

check_valid('torch.as_tensor', generated_inputs['torch.as_tensor_1'], lib="torch")
