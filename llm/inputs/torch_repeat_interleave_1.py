
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def repeat_interleave_inputs():
    list_of_inputs = []

    # To fix the KeyError: 'out', the 'out' key must be included in the dictionary.
    # The TypeError previously encountered suggests an issue with how the arguments are passed to the function by the execution framework.
    # The following inputs strictly adhere to the provided signature {'input': 'tensor', 'repeats': 'integer', 'dim': 'integer', 'out': 'tensor'},
    # including correctly-shaped and typed 'out' tensors, which is the correct way to fix the KeyError.

    # Case 1: Simple 1D tensor, repeat along dim 0
    input_np = np.array([1, 2, 3], dtype=np.int32)
    repeats = 2
    dim = 0
    out_np = np.empty(6, dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))

    # Case 2: 2D tensor, repeat along dim 0
    input_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    repeats = 3
    dim = 0
    out_np = np.empty((6, 2), dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))

    # Case 3: 2D tensor, repeat along dim 1
    input_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
    repeats = 2
    dim = 1
    out_np = np.empty((2, 4), dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))
    
    # Case 4: 2D tensor, negative dimension
    input_np = np.array([[10, 20], [30, 40]], dtype=np.int64)
    repeats = 2
    dim = -1
    out_np = np.empty((2, 4), dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))

    # Case 5: 3D tensor, repeat along middle dimension
    input_np = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    repeats = 2
    dim = 1
    out_np = np.empty((2, 4, 2), dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))

    # Case 6: Repeats = 1 (no change in size)
    input_np = np.array([10, 20, 30], dtype=np.int16)
    repeats = 1
    dim = 0
    out_np = np.empty(3, dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))

    # Case 7: Repeats = 0, results in an empty tensor along the specified dim
    input_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    repeats = 0
    dim = 1
    out_np = np.empty((2, 0), dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))

    # Case 8: 3D tensor, repeat along last dimension
    input_np = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    repeats = 2
    dim = 2
    out_np = np.empty((2, 2, 6), dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))

    # Case 9: Single element tensor
    input_np = np.array([100], dtype=np.int64)
    repeats = 5
    dim = 0
    out_np = np.empty(5, dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))
    
    # Case 10: Input with a zero-sized dimension (non-repeating dim)
    input_np = np.zeros((2, 0, 3), dtype=np.float32)
    repeats = 4
    dim = 0
    out_np = np.empty((8, 0, 3), dtype=input_np.dtype)
    list_of_inputs.append(copy.deepcopy({
        "input": input_np,
        "repeats": repeats,
        "dim": dim,
        "out": out_np
    }))

    return list_of_inputs

generated_inputs["torch.repeat_interleave_1"] = repeat_interleave_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.repeat_interleave_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.repeat_interleave_1'.")

check_valid('torch.repeat_interleave', generated_inputs['torch.repeat_interleave_1'], lib="torch", suffix=1)
