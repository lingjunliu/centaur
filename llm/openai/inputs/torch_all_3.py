
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_all_3_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([True, True, False, True, True], dtype=bool)
    dim = (0,)
    keepdim = False
    out = np.empty((), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 2
    input_arr = np.array([[True, True, True, True],
                          [True, False, True, True],
                          [True, True, True, True]], dtype=bool)
    dim = (1,)
    keepdim = False
    out = np.empty((3,), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 3
    input_arr = np.array([[1, 2, 3, 4],
                          [0, 5, 6, 7]], dtype=np.int64)
    dim = (0,)
    keepdim = True
    out = np.empty((1, 4), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 4
    input_arr = np.arange(2*3*4, dtype=np.float32).reshape(2, 3, 4)
    dim = (2,)
    keepdim = False
    out = np.empty((2, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 5 (uint8 result dtype uint8)
    input_arr = np.array([[[1, 0, 2],
                           [3, 4, 0]],
                          [[1, 1, 1],
                           [1, 0, 1]]], dtype=np.uint8)
    dim = (0, 2)
    keepdim = True
    out = np.empty((1, 2, 1), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 6 (negative dims)
    input_arr = np.array([[[[1, 0, 1, 1]],
                           [[1, 1, 1, 0]],
                           [[1, 1, 1, 1]]],
                          [[[1, 1, 1, 1]],
                           [[-1, 1, 1, 1]],
                           [[1, 1, 1, 1]]]], dtype=np.int8)
    dim = (-1, -3)
    keepdim = False
    out = np.empty((2, 1), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 7 (5D, multiple dims, keepdim True)
    input_arr = np.arange(2*4*3*5*2, dtype=np.float64).reshape(2, 4, 3, 5, 2)
    dim = (1, 3, 4)
    keepdim = True
    out = np.empty((2, 1, 3, 1, 1), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 8 (empty uint8, scalar out uint8)
    input_arr = np.array([], dtype=np.uint8)
    dim = (0,)
    keepdim = False
    out = np.empty((), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 9 (empty dimension, keepdim True)
    input_arr = np.empty((4, 0), dtype=bool)
    dim = (-1,)
    keepdim = True
    out = np.empty((4, 1), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 10 (reduce all dims to scalar)
    input_arr = np.array([[[-1, 2, 3], [4, 5, 6]],
                          [[7, 8, 9], [10, 11, 0]]], dtype=np.int64)
    dim = (0, 1, 2)
    keepdim = False
    out = np.empty((), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 11 (float16, keepdim True)
    input_arr = np.array([[0.0, 1.0, 2.0],
                          [3.0, 0.0, 1.0]], dtype=np.float16)
    dim = (0,)
    keepdim = True
    out = np.empty((1, 3), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 12 (uint8 with dims out-of-order)
    input_arr = np.array([[[1, 1, 0, 1],
                           [1, 1, 1, 1]],
                          [[0, 1, 1, 1],
                           [1, 1, 1, 0]],
                          [[1, 1, 1, 1],
                           [1, 0, 1, 1]]], dtype=np.uint8)
    dim = (2, 0)
    keepdim = False
    out = np.empty((2,), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    return list_of_inputs

generated_inputs["torch.all_3"] = torch_all_3_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.all_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.all_3'.")


check_valid('torch.all', generated_inputs['torch.all_3'], lib="torch", suffix=3)
