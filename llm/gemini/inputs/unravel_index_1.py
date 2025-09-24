
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def unravel_index_inputs():
    list_of_inputs = []

    # Example 1: Basic example with integer indices
    indices = np.array([22, 41, 37]).astype(np.int64)
    shape = (7, 6)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Single index
    indices = np.array(5).astype(np.int64)
    shape = (2, 3)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Multi-dimensional shape
    indices = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64)
    shape = (2, 1, 3)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Larger indices and shape
    indices = np.array([1621, 792, 797, 1509, 629, 1343, 1155, 1324, 1015, 1336]).astype(np.int64)
    shape = (23, 83)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Zero index
    indices = np.array([0]).astype(np.int64)
    shape = (5, 5)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 7: 3D indices and shape
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7]).astype(np.int64)
    shape = (2, 2, 2)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.unravel_index_1"] = unravel_index_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.unravel_index_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unravel_index_1'.")

check_valid('torch.unravel_index', generated_inputs['torch.unravel_index_1'], lib="torch")
