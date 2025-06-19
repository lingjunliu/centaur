
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: 1D int tensor
    input1 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    input_dict1 = {"input": input1, "out": np.empty_like(input1)}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D uint tensor
    input2 = np.array([[255, 0], [128, 64]], dtype=np.uint8)
    input_dict2 = {"input": input2, "out": np.empty_like(input2)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D int tensor with negative values
    input3 = np.array([[[1, -2], [-3, 4]], [[-5, 6], [7, -8]]], dtype=np.int16)
    input_dict3 = {"input": input3, "out": np.empty_like(input3)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D bool tensor
    input4 = np.array([True, False, True, False], dtype=bool)
    input_dict4 = {"input": input4, "out": np.empty_like(input4)}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: scalar input
    input5 = np.array(10, dtype=np.int32)
    input_dict5 = {"input": input5, "out": np.array(0, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.bitwise_not"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bitwise_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_not'.")

check_valid('torch.bitwise_not', generated_inputs['torch.bitwise_not'], lib="torch")
