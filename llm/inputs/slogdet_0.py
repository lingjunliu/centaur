
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def slogdet_inputs():
    list_of_inputs = []

    # Input 1: 2x2 matrix with positive determinant
    input1 = np.array([[2, 0], [0, 3]], dtype=np.float64)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3x3 matrix with negative determinant
    input2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Batch of 2x2 matrices (3D tensor)
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 4x4 matrix with a zero determinant
    input4 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex 2x2 matrix
    input5 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex128)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 1x1 matrix
    input6 = np.array([[5]], dtype=np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.slogdet"] = slogdet_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.slogdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.slogdet'.")

check_valid('torch.slogdet', generated_inputs['torch.slogdet'], lib="torch")
