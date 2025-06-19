
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def rsqrt__inputs():
    list_of_inputs = []

    # Input 1: 1D tensor of positive floats
    input1 = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor of positive floats
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor of positive floats
    input3 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor with a zero value (should not cause error, will result in inf)
    input4 = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor with different values
    input5 = np.array([[0.25, 1.0], [2.25, 4.0]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Large values
    input6 = np.array([100.0, 10000.0, 1000000.0], dtype=np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Small values
    input7 = np.array([0.01, 0.0001, 0.000001], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.rsqrt_"] = rsqrt__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rsqrt_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rsqrt_'.")

check_valid('torch.rsqrt_', generated_inputs['torch.rsqrt_'], lib="torch")
