
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def minimum_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensors
    input1 = np.array([1, 2, -1, 4, 5])
    other1 = np.array([3, 0, 4, -2, 6])
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensors with different shapes
    input2 = np.array([[1.5, 2.0], [3.5, 4.0]])
    other2 = np.array([[2.0, 1.0], [4.0, 3.0]])
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensors with negative and NaN values
    input3 = np.array([-1.0, 2.0, np.nan, 4.0])
    other3 = np.array([3.0, -2.0, 1.0, np.nan])
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multi-dimensional integer tensors
    input4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other4 = np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]])
    input_dict4 = {"input": input4, "other": other4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensors with different dtypes that can be casted
    input5 = np.array([1, 2, 3], dtype=np.int32)
    other5 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Broadcasting example
    input6 = np.array([[1, 2, 3]])
    other6 = np.array([[4], [5], [6]])
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.minimum"] = minimum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.minimum'.")

check_valid('torch.minimum', generated_inputs['torch.minimum'], lib="torch")
