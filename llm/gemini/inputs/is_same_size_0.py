
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def is_same_size_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D tensors with same size
    input1 = np.random.randn(2, 3).astype(np.float32)
    target1 = np.random.randn(2, 3).astype(np.float32)
    input_dict1 = {"input": input1, "target": target1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Different data types, same size
    input2 = np.random.randint(0, 10, size=(5, 4)).astype(np.int64)
    target2 = np.random.randn(5, 4).astype(np.float64)
    input_dict2 = {"input": input2, "target": target2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 3D tensors with same size
    input3 = np.random.randn(1, 5, 5).astype(np.float32)
    target3 = np.random.randn(1, 5, 5).astype(np.float32)
    input_dict3 = {"input": input3, "target": target3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 1D tensors with same size
    input4 = np.random.randn(10).astype(np.float32)
    target4 = np.random.randn(10).astype(np.float32)
    input_dict4 = {"input": input4, "target": target4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Higher dimensions, same size
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    target5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict5 = {"input": input5, "target": target5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Tensors with zero dimension
    input6 = np.random.randn(0, 5).astype(np.float32)
    target6 = np.random.randn(0, 5).astype(np.float32)
    input_dict6 = {"input": input6, "target": target6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: Tensors with negative values
    input7 = np.random.randn(2, 3).astype(np.float32) * -1
    target7 = np.random.randn(2, 3).astype(np.float32) * -1
    input_dict7 = {"input": input7, "target": target7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.is_same_size"] = is_same_size_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.is_same_size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_same_size'.")

check_valid('torch.is_same_size', generated_inputs['torch.is_same_size'], lib="torch")
