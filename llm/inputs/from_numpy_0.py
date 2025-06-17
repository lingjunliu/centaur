
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def from_numpy_inputs():
    list_of_inputs = []

    # Case 1: 1D array of integers
    np_array_1d_int = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    input_dict = {"ndarray": np_array_1d_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D array of floats
    np_array_2d_float = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    input_dict = {"ndarray": np_array_2d_float}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.from_numpy"] = from_numpy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.from_numpy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.from_numpy'.")

check_valid('torch.from_numpy', generated_inputs['torch.from_numpy'], lib="torch")
