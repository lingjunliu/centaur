
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def i0_inputs():
    list_of_inputs = []

    # Input 1: Scalar float
    input1 = np.array(2.5, dtype=np.float32)
    input_dict1 = {"input": torch.tensor(input1)}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D float tensor
    input2 = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    input_dict2 = {"input": torch.tensor(input2)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D float tensor
    input3 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict3 = {"input": torch.tensor(input3)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D float tensor with negative values
    input4 = np.array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]], dtype=np.float32)
    input_dict4 = {"input": torch.tensor(input4)}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Scalar integer
    input5 = np.array(5, dtype=np.int32)
    input_dict5 = {"input": torch.tensor(input5, dtype=torch.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.i0"] = i0_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.i0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.i0'.")

check_valid('torch.i0', generated_inputs['torch.i0'], lib="torch")
