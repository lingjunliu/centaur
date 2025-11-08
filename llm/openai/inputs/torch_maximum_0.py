
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def maximum_inputs():
    list_of_inputs = []

    # Input 1: 1D int64
    input_arr = np.array([1, 2, -1], dtype=np.int64)
    other_arr = np.array([3, 0, 4], dtype=np.int64)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 2: 1D float32 with NaNs
    input_arr = np.array([np.nan, -2.5, 0.0, 3.1], dtype=np.float32)
    other_arr = np.array([0.5, np.nan, -1.0, 2.9], dtype=np.float32)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 3: 2D int32
    input_arr = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.int32)
    other_arr = np.array([[0, 5, -3], [7, 1, 2]], dtype=np.int32)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 4: Broadcast 2D float64 (3,1) vs (1,4)
    input_arr = np.array([[1.5], [-2.0], [0.0]], dtype=np.float64)
    other_arr = np.array([[0.0, -3.0, 2.2, 5.5]], dtype=np.float64)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 5: Scalar int16 vs vector
    input_arr = np.array(-7, dtype=np.int16)
    other_arr = np.array([1, -8, 9], dtype=np.int16)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 6: 3D float16 with broadcast (2,1,3) vs (1,4,1) -> (2,4,3)
    input_arr = np.array([[[1.0, -1.0, 0.5]], [[-0.5, 2.0, -3.0]]], dtype=np.float16)
    other_arr = np.array([[[0.0], [1.5], [-2.0], [3.0]]], dtype=np.float16)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 7: Boolean
    input_arr = np.array([[True, False], [False, True]], dtype=bool)
    other_arr = np.array([[False, False], [True, True]], dtype=bool)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 8: Empty dimension float32 (0,3)
    input_arr = np.empty((0, 3), dtype=np.float32)
    other_arr = np.empty((0, 3), dtype=np.float32)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 9: 4D int64 same shape (2,1,3,1)
    input_arr = np.array([[[[1], [2], [3]]],
                          [[[4], [5], [6]]]], dtype=np.int64)
    other_arr = np.array([[[[0], [10], [-1]]],
                          [[[7], [1], [2]]]], dtype=np.int64)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 10: uint8 1D
    input_arr = np.array([0, 255, 128, 64, 10], dtype=np.uint8)
    other_arr = np.array([5, 128, 200, 32, 10], dtype=np.uint8)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 11: float64 with infinities and signed zeros
    input_arr = np.array([[0.0, -0.0, np.inf], [-np.inf, 3.3, -4.4]], dtype=np.float64)
    other_arr = np.array([[-0.0, 0.0, -np.inf], [np.inf, -3.3, 4.4]], dtype=np.float64)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # Input 12: 3D float32 broadcast (1,3,1) vs (2,1,4) -> (2,3,4)
    input_arr = np.array([[[1.0], [2.0], [-3.0]]], dtype=np.float32)
    other_arr = np.array([[[0.5, -1.0, 2.0, -2.5]],
                          [[-0.5, 1.0, -2.0, 2.5]]], dtype=np.float32)
    out_arr = np.empty(np.broadcast(input_arr, other_arr).shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.maximum"] = maximum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.maximum'.")


check_valid('torch.maximum', generated_inputs['torch.maximum'], lib="torch", suffix=0)
