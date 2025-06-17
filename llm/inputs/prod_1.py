
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_prod_inputs():
    list_of_inputs = []

    # Test case 1: Basic 1D tensor
    input_tensor = np.array([1, 2, 3, 4, 5])
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 2D tensor with dim and keepdim
    input_tensor = np.array([[1, 2], [3, 4]])
    input_dict = {"input": input_tensor, "dim": 1, "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 3: 3D tensor with negative values and dtype
    input_tensor = np.array([[[1, -2], [3, 4]], [[5, 6], [-7, 8]]], dtype=np.float32)
    input_dict = {"input": input_tensor, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Large tensor with dim=0
    input_tensor = np.random.rand(10, 5, 2)
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Tensor with only one element
    input_tensor = np.array([[[5]]])
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Empty tensor
    input_tensor = np.array([])
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Integer tensor
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 8: Bool tensor
    input_tensor = np.array([[True, False], [True, True]])
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.prod_1"] = torch_prod_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.prod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.prod_1'.")

check_valid('torch.prod', generated_inputs['torch.prod_1'], lib="torch")
