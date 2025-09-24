
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_qr_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    input1 = np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    input_dict1 = {"input": input1, "some": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Non-square matrix (m > n)
    input2 = np.array([[1., 2], [3, 4], [5, 6]], dtype=np.float32)
    input_dict2 = {"input": input2, "some": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Non-square matrix (m < n)
    input3 = np.array([[1., 2, 3], [4, 5, 6]], dtype=np.float64)
    input_dict3 = {"input": input3, "some": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Batch of matrices
    input4 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict4 = {"input": input4, "some": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Single element matrix
    input5 = np.array([[5.0]])
    input_dict5 = {"input": input5, "some": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Matrix with zeros
    input6 = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
    input_dict6 = {"input": input6, "some": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Matrix with negative values
    input7 = np.array([[-1., 2], [3, -4]], dtype=np.float32)
    input_dict7 = {"input": input7, "some": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.qr"] = torch_qr_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.qr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.qr'.")

check_valid('torch.qr', generated_inputs['torch.qr'], lib="torch")
