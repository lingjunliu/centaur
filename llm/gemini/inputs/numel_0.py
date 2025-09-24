
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def numel_inputs():
    list_of_inputs = []

    input1 = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.zeros((4, 4)).astype(np.int64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1.1, 2.2], [3.3, 4.4]]).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int8)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-1, -2], [-3, -4]]).astype(np.int16)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1+1j, 2+2j, 3+3j]).astype(np.complex64)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.numel"] = numel_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.numel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.numel'.")

check_valid('torch.numel', generated_inputs['torch.numel'], lib="torch")
