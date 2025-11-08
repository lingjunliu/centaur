
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def any_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([False, True, False, False, False], dtype=np.bool_)
    dim = np.int64(0)
    keepdim = False
    out = np.empty((), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 2
    input_arr = np.array([[0, 1, 0],
                          [2, 0, 3]], dtype=np.int64)
    dim = np.int32(1)
    keepdim = False
    out = np.empty((2,), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 3
    input_arr = np.array([[-1.0, 0.0, 2.5, 0.0],
                          [0.0, 0.0, -3.14, 4.2],
                          [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    dim = np.int64(0)
    keepdim = True
    out = np.empty((1, 4), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 4 (uint8, output dtype uint8)
    input_arr = np.array([[[0, 0, 1, 0],
                           [0, 0, 0, 0],
                           [2, 0, 0, 0]],
                          [[0, 3, 0, 0],
                           [0, 0, 0, 4],
                           [0, 0, 0, 0]]], dtype=np.uint8)
    dim = np.int32(-1)
    keepdim = False
    out = np.empty((2, 3), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 5
    input_arr = (np.random.randn(2, 3, 5, 4).astype(np.float16) * 2 - 1)
    dim = np.int64(2)
    keepdim = False
    out = np.empty((2, 3, 4), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 6
    input_arr = np.array([0], dtype=np.int8)
    dim = np.int32(0)
    keepdim = True
    out = np.empty((1,), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 7
    input_arr = np.array([[[True, False, True]],
                          [[False, False, False]],
                          [[True, True, True]],
                          [[False, True, False]]], dtype=np.bool_)
    dim = np.int64(1)
    keepdim = True
    out = np.empty((4, 1, 3), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 8
    input_arr = np.random.randint(-1, 2, size=(2, 2, 2, 2, 2), dtype=np.int32)
    dim = np.int32(-3)
    keepdim = False
    out = np.empty((2, 2, 2, 2), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 9 (uint8, all zeros -> output uint8)
    input_arr = np.zeros((3, 3), dtype=np.uint8)
    dim = np.int64(0)
    keepdim = True
    out = np.empty((1, 3), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 10 (complex64)
    input_arr = (np.array([[[0+0j, 1+0j],
                            [0+0j, 0+0j],
                            [0+2j, 0+0j]],
                           [[0+0j, 0+0j],
                            [3+0j, 0+0j],
                            [0+0j, 0+0j]]], dtype=np.complex64))
    dim = np.int32(2)
    keepdim = True
    out = np.empty((2, 3, 1), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 11
    input_arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dim = np.int64(-1)
    keepdim = False
    out = np.empty((), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    # Input 12
    input_arr = np.array([[[[np.nan],
                             [0.0],
                             [np.inf],
                             [-0.0]]],
                           [[[1.0],
                             [-2.0],
                             [0.0],
                             [3.0]]],
                           [[[0.0],
                             [0.0],
                             [0.0],
                             [0.0]]]], dtype=np.float64).reshape(3, 1, 4, 1)
    dim = np.int32(0)
    keepdim = True
    out = np.empty((1, 1, 4, 1), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }))

    return list_of_inputs

generated_inputs["torch.any_2"] = any_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.any_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.any_2'.")


check_valid('torch.any', generated_inputs['torch.any_2'], lib="torch", suffix=2)
