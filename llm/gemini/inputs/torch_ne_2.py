
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def ne_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor vs. float
    input = np.array([1, 2, 3], dtype=np.int32)
    other = 2.0
    out = np.array([False, False, False], dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor vs. float
    input = np.array([[1, 2], [3, 4]], dtype=np.float32)
    other = 3.0
    out = np.array([[False, False], [False, False]], dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    input = np.array([-1, 0, 1], dtype=np.int64)
    other = 0.0
    out = np.array([False, False, False], dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Larger tensor
    input = np.random.rand(5, 5).astype(np.float64)
    other = 0.5
    out = np.zeros((5, 5), dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zeros
    input = np.zeros((3, 2), dtype=np.float32)
    other = 0.0
    out = np.array([[False, False], [False, False], [False, False]], dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Mixed positive and negative
    input = np.array([[-1, 1], [-2, 2]], dtype=np.int32)
    other = 1.0
    out = np.array([[False, False], [False, False]], dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor
    input = np.random.rand(2, 3, 4).astype(np.float32)
    other = 0.2
    out = np.zeros((2, 3, 4), dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different data type
    input = np.array([1, 2, 3], dtype=np.uint8)
    other = 2.0
    out = np.array([False, False, False], dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large float value
    input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other = 1e9
    out = np.array([False, False, False], dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small float value
    input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other = 1e-9
    out = np.array([False, False, False], dtype=bool)
    input_dict = {"input": input, "other": other, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.ne_2"] = ne_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ne_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ne_2'.")

check_valid('torch.ne', generated_inputs['torch.ne_2'], lib="torch", suffix=2)
