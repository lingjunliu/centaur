
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def unique_consecutive_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensor
    input_tensor = np.array([1, 1, 2, 2, 3, 1, 1, 4], dtype=np.int64)
    input_dict = {"input": input_tensor, "return_inverse": False, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensor
    input_tensor = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0, 4.0], dtype=np.float32)
    input_dict = {"input": input_tensor, "return_inverse": True, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values
    input_tensor = np.array([-1, -1, 0, 1, 1, 1, -2, -2], dtype=np.int32)
    input_dict = {"input": input_tensor, "return_inverse": False, "return_counts": True, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Multi-dimensional tensor (2D)
    input_tensor = np.array([[1, 1, 2], [2, 3, 3], [3, 3, 4]], dtype=np.int16)
    input_dict = {"input": input_tensor, "return_inverse": True, "return_counts": True, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Multi-dimensional tensor (3D)
    input_tensor = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=np.int8)
    input_dict = {"input": input_tensor, "return_inverse": False, "return_counts": False, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Empty Tensor
    input_tensor = np.array([], dtype=np.int64)
    input_dict = {"input": input_tensor, "return_inverse": False, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.unique_consecutive_2"] = unique_consecutive_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.unique_consecutive_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unique_consecutive_2'.")

check_valid('torch.unique_consecutive', generated_inputs['torch.unique_consecutive_2'], lib="torch")
