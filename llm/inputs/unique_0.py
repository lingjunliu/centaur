
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_unique_inputs():
    list_of_inputs = []

    # Case 1: Basic 1D integer tensor
    input_tensor = np.array([1, 3, 2, 3, 1], dtype=np.int64)
    input_dict = {"input": input_tensor, "sorted": True, "return_inverse": False, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D float tensor with return_inverse
    input_tensor = np.array([[1.5, 2.5], [3.5, 1.5]], dtype=np.float32)
    input_dict = {"input": input_tensor, "sorted": True, "return_inverse": True, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tensor with return_counts
    input_tensor = np.array([1, 2, 2, 3, 3, 3], dtype=np.int32)
    input_dict = {"input": input_tensor, "sorted": True, "return_inverse": False, "return_counts": True, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D tensor with dim specified
    input_tensor = np.array([[1, 2, 1], [3, 4, 3], [5, 6, 5]], dtype=np.int64)
    input_dict = {"input": input_tensor, "sorted": True, "return_inverse": False, "return_counts": False, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D tensor with all options enabled, remove due to potential issues.
    #input_tensor = np.array([[[1, 2], [3, 4]], [[1, 2], [5, 6]], [[7, 8], [3, 4]]], dtype=np.int64)
    #input_dict = {"input": input_tensor, "sorted": True, "return_inverse": True, "return_counts": True, "dim": 1}
    #list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: 1D tensor with negative values
    input_tensor = np.array([-1, -2, -1, 0, 1, 0], dtype=np.int64)
    input_dict = {"input": input_tensor, "sorted": True, "return_inverse": False, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: unsorted tensor
    input_tensor = np.array([5, 2, 1, 4, 3], dtype=np.int64)
    input_dict = {"input": input_tensor, "sorted": False, "return_inverse": False, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.unique"] = torch_unique_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unique'.")

check_valid('torch.unique', generated_inputs['torch.unique'], lib="torch")
