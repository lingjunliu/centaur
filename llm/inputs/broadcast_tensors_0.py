
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def broadcast_tensors_inputs():
    list_of_inputs = []

    # Case 1: Basic broadcasting
    tensors = [torch.tensor(np.array([1, 2, 3])), torch.tensor(np.array([[4], [5]]))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different data types
    tensors = [torch.tensor(np.array([1, 2, 3], dtype=np.int32)), torch.tensor(np.array([4.0], dtype=np.float64))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Higher dimensions
    tensors = [torch.tensor(np.array([[1, 2], [3, 4]])), torch.tensor(np.array([5, 6]))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Complex numbers
    tensors = [torch.tensor(np.array([1+1j, 2+2j])), torch.tensor(np.array([3+3j]))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Broadcasting with size 1 dimensions
    tensors = [torch.tensor(np.array([[1], [2]])), torch.tensor(np.array([3, 4]))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.broadcast_tensors"] = broadcast_tensors_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.broadcast_tensors' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.broadcast_tensors'.")

check_valid('torch.broadcast_tensors', generated_inputs['torch.broadcast_tensors'], lib="torch")
