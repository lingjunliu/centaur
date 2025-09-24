
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def reciprocal__inputs():
    list_of_inputs = []

    # Test case 1: Positive float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Negative float tensor
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Zero value
    input3 = np.array([0.0], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Multi-dimensional float tensor
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Mixed positive and negative floats
    input5 = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: Larger multi-dimensional tensor
    input6 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: Tensor with very small values
    input7 = np.array([1e-8, 2e-7, 3e-9], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.reciprocal_"] = reciprocal__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.reciprocal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.reciprocal_'.")

check_valid('torch.reciprocal_', generated_inputs['torch.reciprocal_'], lib="torch")
