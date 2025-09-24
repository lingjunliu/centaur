
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def det_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 float matrix
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3x3 float matrix
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4x4 complex matrix
    input3 = np.array([[1+1j, 2+0j, 3+0j, 4+0j],
                       [5+0j, 6+1j, 7+0j, 8+0j],
                       [9+0j, 10+0j, 11+1j, 12+0j],
                       [13+0j, 14+0j, 15+0j, 16+1j]], dtype=np.complex64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2x2 float matrix with negative values
    input4 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 5x5 random float matrix
    input5 = np.random.rand(5, 5).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.det"] = det_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.det' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.det'.")

check_valid('torch.det', generated_inputs['torch.det'], lib="torch")
