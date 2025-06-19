
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def unique_consecutive_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer tensor
    input1 = np.array([1, 1, 2, 2, 3, 1, 1, 2]).astype(np.int64)
    input_dict1 = {"input": input1, "return_inverse": False, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor with return_inverse and return_counts
    input2 = np.array([1, 1, 2, 2, 3, 1, 1, 2]).astype(np.int32)
    input_dict2 = {"input": input2, "return_inverse": True, "return_counts": True, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, specifying dimension
    input3 = np.array([[1, 1, 2, 2], [3, 1, 1, 2]]).astype(np.int8)
    input_dict3 = {"input": input3, "return_inverse": False, "return_counts": False, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, specifying dimension, floats
    input4 = np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 1.0], [1.0, 2.0]]]).astype(np.float32)
    input_dict4 = {"input": input4, "return_inverse": False, "return_counts": False, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 1D with negative values
    input5 = np.array([-1, -1, 0, 0, 1, -1, -1, 0]).astype(np.int64)
    input_dict5 = {"input": input5, "return_inverse": False, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D with negative values
    input6 = np.array([[-1, -1, 0, 0], [1, -1, -1, 0]]).astype(np.int64)
    input_dict6 = {"input": input6, "return_inverse": False, "return_counts": False, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Empty tensor
    input7 = np.array([]).astype(np.int64)
    input_dict7 = {"input": input7, "return_inverse": False, "return_counts": False, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.unique_consecutive_1"] = unique_consecutive_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.unique_consecutive_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unique_consecutive_1'.")

check_valid('torch.unique_consecutive', generated_inputs['torch.unique_consecutive_1'], lib="torch")
