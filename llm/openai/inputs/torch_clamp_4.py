
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clamp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, scalar-like min/max (1,)
    input_arr = np.array([-1.7120, 0.1734, -0.0478, -0.0922], dtype=np.float32)
    min_arr = np.array([-0.5], dtype=np.float32)
    max_arr = np.array([0.5], dtype=np.float32)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 2: 2D float64, broadcast min (2,1) and max (1,3)
    input_arr = np.array([[-1.0, 0.5, 2.0],
                          [3.3, -4.4, 0.0]], dtype=np.float64)
    min_arr = np.array([[-0.5],
                        [0.0]], dtype=np.float64)
    max_arr = np.array([[0.5, 1.0, 2.5]], dtype=np.float64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 3: 3D int32, scalar-like min/max via (1,) broadcasting
    input_arr = np.arange(24, dtype=np.int32).reshape(2, 3, 4) - 6
    min_arr = np.array([-2], dtype=np.int32)
    max_arr = np.array([2], dtype=np.int32)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 4: 1D int64, elementwise min/max
    input_arr = np.array([-10, -1, 0, 4, 10], dtype=np.int64)
    min_arr = np.array([-5, -2, 0, 3, 8], dtype=np.int64)
    max_arr = np.array([-3, 0, 1, 5, 9], dtype=np.int64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 5: Empty 2D float32, broadcast along last dim
    input_arr = np.empty((0, 3), dtype=np.float32)
    min_arr = np.array([[0.0, -1.0, 0.5]], dtype=np.float32)
    max_arr = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 6: 3D float16, elementwise min and scalar-like max via (1,1,1)
    input_arr = np.array([[[-1.2, 0.1]],
                          [[2.3, -0.7]],
                          [[0.0, 1.5]]], dtype=np.float16)
    min_arr = np.array([[[-0.5, -0.5]],
                        [[0.0, -1.0]],
                        [[-0.2, 0.2]]], dtype=np.float16)
    max_arr = np.array([[[1.0]]], dtype=np.float16)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 7: 2D uint8, scalar-like min/max
    input_arr = (np.arange(16, dtype=np.uint8).reshape(4, 4) * 16).astype(np.uint8)
    min_arr = np.array([0], dtype=np.uint8)
    max_arr = np.array([200], dtype=np.uint8)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 8: 4D float32, min varies along dim0, max varies along dim1
    input_arr = np.linspace(-3.0, 3.0, num=16, dtype=np.float32).reshape(2, 2, 2, 2)
    min_arr = np.array([-1.0, 0.0], dtype=np.float32).reshape(2, 1, 1, 1)
    max_arr = np.array([0.0, 1.0], dtype=np.float32).reshape(1, 2, 1, 1)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 9: 1D float64 with -inf, inf, nan
    input_arr = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan], dtype=np.float64)
    min_arr = np.array([-1.0], dtype=np.float64)
    max_arr = np.array([1.0], dtype=np.float64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 10: 1D float32, min > max everywhere
    input_arr = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    min_arr = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    max_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 11: 2D int16, broadcast min/max
    input_arr = np.array([[-10, 0, 10],
                          [5, -5, 15]], dtype=np.int16)
    min_arr = np.array([[-5],
                        [0]], dtype=np.int16)
    max_arr = np.array([[0, 10, 5]], dtype=np.int16)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    # Input 12: 0-D scalar float32
    input_arr = np.array(3.14, dtype=np.float32)
    min_arr = np.array(0.0, dtype=np.float32)
    max_arr = np.array(1.0, dtype=np.float32)
    out_arr = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_arr, "max": max_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.clamp_4"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clamp_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_4'.")


check_valid('torch.clamp', generated_inputs['torch.clamp_4'], lib="torch", suffix=4)
