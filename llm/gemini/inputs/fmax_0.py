
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def fmax_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input1 = np.array([9.7, float('nan'), 3.1, -2.5]).astype(np.float32)
    input2 = np.array([-2.2, 0.5, float('nan'), 1.0]).astype(np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Integer tensors
    input1 = np.array([1, 5, -2, 0]).astype(np.int32)
    input2 = np.array([3, -1, 4, 2]).astype(np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Broadcasting with scalar
    input1 = np.array([1.0, 2.0, 3.0]).astype(np.float64)
    input2 = np.array(2.0).astype(np.float64)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Multi-dimensional tensors
    input1 = np.array([[1, 2], [3, float('nan')]]).astype(np.float32)
    input2 = np.array([[4, float('nan')], [float('nan'), 6]]).astype(np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 5: tensors with negative values and zeros
    input1 = np.array([-1.0, 0.0, 1.0]).astype(np.float32)
    input2 = np.array([0.0, -2.0, 2.0]).astype(np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Larger tensors
    input1 = np.random.randn(5, 5).astype(np.float32)
    input2 = np.random.randn(5, 5).astype(np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Mixed positive and negative, including NaN
    input1 = np.array([1.0, -2.0, float('nan'), 4.0, -5.0]).astype(np.float64)
    input2 = np.array([-1.0, 2.0, 3.0, float('nan'), 5.0]).astype(np.float64)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.fmax"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fmax'.")

check_valid('torch.fmax', generated_inputs['torch.fmax'], lib="torch")
