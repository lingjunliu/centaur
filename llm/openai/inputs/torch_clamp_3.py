
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clamp_inputs():
    list_of_inputs = []

    # 1: 1D float32, scalar max (numpy scalar) and float min
    input_arr = np.array([-1.2, 0.3, 1.5, -0.7, 2.2], dtype=np.float32)
    min_val = -0.5
    max_arr = np.float32(0.8)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    # 2: 2D float64
    input_arr = np.array([[1.0, -2.0, 3.5],
                          [4.2, -5.1, 0.0]], dtype=np.float64)
    min_val = -1.0
    max_arr = np.float64(2.5)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    # 3: 3D float16
    input_arr = np.array([[[ -2.0,  0.5],
                           [  3.2, -4.1]],
                          [[ 10.0, -0.3],
                           [  1.1,  2.2]]], dtype=np.float16)
    min_val = 0.0
    max_arr = np.float16(1.0)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    # 4: 4D float32
    input_arr = np.random.randn(1, 3, 2, 2).astype(np.float32)
    min_val = -2.0
    max_arr = np.float32(0.0)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    # 5: Empty input (0,3), float32
    input_arr = np.empty((0, 3), dtype=np.float32)
    min_val = -1.0
    max_arr = np.float32(2.0)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    # 6: 1D float64 with NaN/Inf
    input_arr = np.array([np.nan, -1e9, 0.0, 1e9, np.inf, -np.inf], dtype=np.float64)
    min_val = -1e9
    max_arr = np.float64(1e9)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    # 7: 2D float32, positive clamp
    input_arr = np.array([[0.1, 100.0, 10000.0],
                          [-50.0, 500.0, 1.0]], dtype=np.float32)
    min_val = 0.0
    max_arr = np.float32(1000.0)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    # 8: 2D float32, negative upper bound
    input_arr = np.array([[-10.0, -2.0, -0.5],
                          [-3.3, -1.1, -7.7]], dtype=np.float32)
    min_val = -5.0
    max_arr = np.float32(-1.0)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    # 9: 3D float32, min == max clamp to constant
    input_arr = np.array([[[ -2.0], [ 0.0], [ 2.0]],
                          [[  1.0], [-1.5], [ 3.3]]], dtype=np.float32)
    min_val = 2.0
    max_arr = np.float32(2.0)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    # 10: 1D float32 zeros
    input_arr = np.zeros((5,), dtype=np.float32)
    min_val = 0.0
    max_arr = np.float32(0.0)
    out_arr = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "min": min_val, "max": max_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.clamp_3"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clamp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_3'.")


check_valid('torch.clamp', generated_inputs['torch.clamp_3'], lib="torch", suffix=3)
