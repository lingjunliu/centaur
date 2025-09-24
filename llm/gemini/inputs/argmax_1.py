
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def argmax_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor with positive values
    input_1 = np.array([1.0, 3.0, 2.0, 4.0]).astype(np.float32)
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 2D tensor with negative values
    input_2 = np.array([[-1.0, -3.0, -2.0], [-4.0, -5.0, -6.0]]).astype(np.float32)
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 3D tensor with mixed positive and negative values
    input_3 = np.array([[[1, -2], [3, 4]], [[-5, 6], [7, -8]]]).astype(np.float32)
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 1D tensor with integer values
    input_4 = np.array([1, 3, 2, 4]).astype(np.int64)
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Case 5: 2D tensor with different data type (int32)
    input_5 = np.array([[5, 2, 9], [1, 7, 3]]).astype(np.int32)
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Tensor with all equal values
    input_6 = np.array([[2, 2, 2], [2, 2, 2]]).astype(np.float32)
    input_dict_6 = {"input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.argmax_1"] = argmax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.argmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argmax_1'.")

check_valid('torch.argmax', generated_inputs['torch.argmax_1'], lib="torch")
