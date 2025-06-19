
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_le_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensors
    input1 = np.array([[1, 2], [3, 4]])
    input2 = np.array([[1, 1], [4, 4]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensors with broadcasting
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = 2.5
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values and different shapes
    input1 = np.array([[-1, 0, 1], [-2, -1, 0]])
    input2 = np.array([-1, -1, -1])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tensors
    input1 = np.array([1, 2, 3, 4, 5])
    input2 = np.array([5, 4, 3, 2, 1])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Broadcasting with a scalar
    input1 = np.array([[1, 2, 3], [4, 5, 6]])
    input2 = 4
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 3D tensors
    input1 = np.random.rand(2, 3, 4)
    input2 = np.random.rand(2, 3, 4)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Mixed integer and float
    input1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input2 = np.array([[1.5, 1.5], [3.5, 4.5]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.le"] = torch_le_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.le' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.le'.")

check_valid('torch.le', generated_inputs['torch.le'], lib="torch")
