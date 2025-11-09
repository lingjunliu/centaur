
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def bitwise_left_shift_inputs():
    list_of_inputs = []
    
    input_val = np.array([1, 2, 3, 4], dtype=np.int32)
    other = np.array([1, 2, 3, 4], dtype=np.int32)
    out = np.empty_like(input_val)
    list_of_inputs.append(copy.deepcopy({
        "input": input_val,
        "other": other,
        "out": out
    }))
    
    input_val = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other = np.array([[1, 1], [2, 2]], dtype=np.int32)
    out = np.empty_like(input_val)
    list_of_inputs.append(copy.deepcopy({
        "input": input_val,
        "other": other,
        "out": out
    }))
    
    input_val = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    other = np.array([[[1, 2], [1, 2]], [[1, 1], [2, 2]]], dtype=np.int32)
    out = np.empty_like(input_val)
    list_of_inputs.append(copy.deepcopy({
        "input": input_val,
        "other": other,
        "out": out
    }))
    
    input_val = np.array([10, 20, 30], dtype=np.int64)
    other = np.array([1, 2, 3], dtype=np.int64)
    out = np.empty_like(input_val)
    list_of_inputs.append(copy.deepcopy({
        "input": input_val,
        "other": other,
        "out": out
    }))
    
    input_val = np.array([1, 2, 3, 4], dtype=np.int8)
    other = np.array([1, 1, 2, 2], dtype=np.int8)
    out = np.empty_like(input_val)
    list_of_inputs.append(copy.deepcopy({
        "input": input_val,
        "other": other,
        "out": out
    }))
    
    input_val = np.array([100, 200, 300], dtype=np.int16)
    other = np.array([2, 3, 4], dtype=np.int16)
    out = np.empty_like(input_val)
    list_of_inputs.append(copy.deepcopy({
        "input": input_val,
        "other": other,
        "out": out
    }))
    
    input_val = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    other = np.array([2], dtype=np.int32)
    out = np.empty_like(input_val)
    list_of_inputs.append(copy.deepcopy({
        "input": input_val,
        "other": other,
        "out": out
    }))
    
    input_val = np.array([10, 20, 30], dtype=np.int32)
    other = np.array([0, 0, 0], dtype=np.int32)
    out = np.empty_like(input_val)
    list_of_inputs.append(copy.deepcopy({
        "input": input_val,
        "other": other,
        "out": out
    }))
    
    input_val = np.array([5], dtype=np.int32)
    other = np.array([3], dtype=np.int32)
    out = np.empty_like(input_val)
    list_of_inputs.append(copy.deepcopy({
        "input": input_val,
        "other": other,
        "out": out

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bitwise_left_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_left_shift'.")


check_valid('torch.bitwise_left_shift', generated_inputs['torch.bitwise_left_shift'], lib="torch", suffix=0)
