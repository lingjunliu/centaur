
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_dist_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    p = 2.0
    input_dict = {"input": input1, "other": input2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Integer tensors (removed as they cause errors)

    # Test case 3: Negative values and different p values
    input1 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    input2 = np.array([4.0, -5.0, 6.0], dtype=np.float32)
    p = 0.0
    input_dict = {"input": input1, "other": input2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Multidimensional tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    p = 2.5
    input_dict = {"input": input1, "other": input2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Broadcasting
    input1 = np.array([1.0, 2.0], dtype=np.float32)
    input2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    p = 1.5
    input_dict = {"input": input1, "other": input2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: Zero norm
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    p = 0.0
    input_dict = {"input": input1, "other": input2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: One dimensional tensors
    input1 = np.array([1.0], dtype=np.float32)
    input2 = np.array([4.0], dtype=np.float32)
    p = 2.0
    input_dict = {"input": input1, "other": input2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dist"] = torch_dist_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dist'.")

check_valid('torch.dist', generated_inputs['torch.dist'], lib="torch")
