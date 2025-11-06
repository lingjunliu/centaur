
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def le_inputs():
    list_of_inputs = []

    # 1: 1D int32 array, positive values
    input = np.array([1, 2, 3, 4], dtype=np.int32)
    other = 2.0
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 2: 2D float32 array with negatives
    input = np.array([[-1.5, 0.0, 1.5],
                      [2.5, -3.2, 4.8]], dtype=np.float32)
    other = -1.0
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 3: 3D int64 array with negatives and positives
    input = np.array([[[-5, 0], [10, -2]],
                      [[3, 7], [-8, 1]]], dtype=np.int64)
    other = 0.0
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 4: 0-D scalar int32
    input = np.array(-3, dtype=np.int32)
    other = -3.0
    out = np.empty((), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 5: Empty 1D float64 array
    input = np.array([], dtype=np.float64)
    other = 0.0
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 6: 4D float16 array
    input = np.arange(2*3*4*5, dtype=np.float16).reshape(2, 3, 4, 5)
    other = 10.0
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 7: Non-contiguous 1D int8 view (stride > 1)
    base = np.arange(10, dtype=np.int8)
    input = base[::2]
    other = 4.0
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 8: 2D float64 with NaN and Inf
    input = np.array([[np.nan, np.inf, -np.inf],
                      [5.0, -2.0, 0.0]], dtype=np.float64)
    other = 1.0
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 9: uint8 array
    input = np.array([0, 127, 128, 255], dtype=np.uint8)
    other = 128.5
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 10: Fortran-ordered 2D float64 array
    input = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
    other = 2.5
    out = np.empty(input.shape, dtype=bool, order='F')
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 11: Empty middle dimension (2,0,3)
    input = np.empty((2, 0, 3), dtype=np.float32)
    other = -0.5
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 12: Large magnitude float64 values
    input = np.array([1e20, -1e20, 0.0, 3.14], dtype=np.float64)
    other = 0.0
    out = np.empty(input.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.le_2"] = le_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.le_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.le_2'.")


check_valid('torch.le', generated_inputs['torch.le_2'], lib="torch", suffix=2)
