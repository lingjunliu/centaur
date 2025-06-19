
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def trunc__inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor with positive and negative floats
    input1 = np.array([1.5, -2.3, 0.7, -0.2, 3.9], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with different float values
    input2 = np.array([[1.1, 2.2, 3.3], [-4.4, -5.5, -6.6]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with a mix of positive and negative values, various magnitudes
    input3 = np.array([[[10.2, -20.3], [30.4, -40.5]], [[-50.6, 60.7], [-70.8, 80.9]]], dtype=np.float16)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with zeros, positive and negative values
    input4 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-3.0, 0.0, 3.0]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger tensor with more varied values
    input5 = np.random.randn(4, 4, 4).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 1D array of very large numbers
    input6 = np.array([1e9 + 0.5, -1e9 - 0.5, 1e9 - 0.5, -1e9 + 0.5], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 2D array with near-integer values
    input7 = np.array([[0.9999, 1.0001], [-1.0001, -0.9999]], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.trunc_"] = trunc__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.trunc_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.trunc_'.")

check_valid('torch.trunc_', generated_inputs['torch.trunc_'], lib="torch")
