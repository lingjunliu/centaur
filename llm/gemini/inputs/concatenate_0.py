
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def concatenate_inputs():
    list_of_inputs = []

    # Case 1: Basic concatenation along axis 0
    tensors1 = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis1 = 0
    input_dict1 = {"tensors": tensors1, "axis": axis1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Concatenation along axis 1
    tensors2 = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    axis2 = 1
    input_dict2 = {"tensors": tensors2, "axis": axis2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Concatenation of tensors with different shapes
    tensors3 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    axis3 = 0
    input_dict3 = {"tensors": tensors3, "axis": axis3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Concatenation of 3D tensors
    tensors4 = [np.random.rand(2, 3, 4), np.random.rand(2, 3, 4)]
    axis4 = 1
    input_dict4 = {"tensors": tensors4, "axis": axis4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Concatenation with negative axis
    tensors5 = [np.random.rand(2, 3, 4), np.random.rand(2, 3, 4)]
    axis5 = -1
    input_dict5 = {"tensors": tensors5, "axis": axis5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.concatenate"] = concatenate_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.concatenate' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.concatenate'.")

check_valid('torch.concatenate', generated_inputs['torch.concatenate'], lib="torch")
