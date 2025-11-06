
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def sgn_inputs():
    list_of_inputs = []

    # 1: 1D float32 array with negatives, zeros, positives
    input_arr = np.array([-3.5, -0.0, 0.0, 2.5, 7.0], dtype=np.float32)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 2: 2D int64 array
    input_arr = np.array([[-3, -1, 0, 1, 5],
                          [10, 0, -7, 2, -2]], dtype=np.int64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 3: 1D complex64 array
    input_arr = np.array([3+4j, 0+0j, -1+1j, 5-12j], dtype=np.complex64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 4: 2D complex128 array
    input_arr = np.array([[1+0j, 0+0j, -2-3j],
                          [7-24j, 0+5j, -0-0j]], dtype=np.complex128)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 5: 3D float64 array with inf and nan
    input_arr = np.array([[[np.nan, -np.inf], [np.inf, -1.0]],
                          [[0.0, 1.0], [-0.0, 3.14]]], dtype=np.float64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 6: 0-D scalar float32
    input_arr = np.array(0.0, dtype=np.float32)
    out_arr = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 7: 4D int32 array
    input_arr = np.array([[[[1, -1], [0, 2]],
                           [[-3, 4], [0, 0]]],
                          [[[5, -6], [7, -8]],
                           [[0, 9], [-10, 0]]]], dtype=np.int32)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 8: Non-contiguous slice float64
    base = np.arange(24, dtype=np.float64).reshape(4, 6) - 12.0
    input_arr = base[:, ::2]
    out_arr = np.empty(input_arr.shape, dtype=input_arr.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 9: Complex64 with very small magnitudes and zero
    input_arr = np.array([1e-12 + 1e-12j, 0+0j, -1e-20 + 0j, 3e-8 - 4e-8j], dtype=np.complex64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 10: Float64 with +0.0 and -0.0
    input_arr = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 11: 3D complex64 array
    input_arr = np.array([[[1+2j, -2+1j],
                           [0+0j, 4-3j]],
                          [[-5+0j, 0-6j],
                           [7+7j, -8-8j]]], dtype=np.complex64)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 12: 2D float16 array
    input_arr = np.array([[0.0, -1.5, 2.25],
                          [-3.5, 0.0, 4.0]], dtype=np.float16)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 13: 1D int8 array
    input_arr = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    # 14: Complex128 with NaN and Inf components
    input_arr = np.array([np.nan + 0j, np.inf + 1j, -np.inf - np.inf*1j, 1 - np.inf*1j], dtype=np.complex128)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.sgn"] = sgn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sgn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sgn'.")


check_valid('torch.sgn', generated_inputs['torch.sgn'], lib="torch", suffix=0)
