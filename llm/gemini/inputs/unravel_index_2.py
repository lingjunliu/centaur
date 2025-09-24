
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def unravel_index_inputs():
    list_of_inputs = []

    # Input 1: Basic case with 1D indices and 2D shape
    indices = np.array([22, 41, 37]).astype(np.int64)
    shape = (7, 6)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D indices and 3D shape
    indices = np.array([[0, 1], [2, 3]]).astype(np.int64)
    shape = (2, 2, 2)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single element indices
    indices = np.array(5).astype(np.int64)
    shape = (2, 3)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger shape and indices, different data type
    indices = np.array([100, 200, 300, 400]).astype(np.int64)
    shape = (10, 20, 30)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Multi-dimensional indices and shape
    indices = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int64)
    shape = (2, 2, 2, 2)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.unravel_index_2"] = unravel_index_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.unravel_index_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unravel_index_2'.")

check_valid('torch.unravel_index', generated_inputs['torch.unravel_index_2'], lib="torch")
