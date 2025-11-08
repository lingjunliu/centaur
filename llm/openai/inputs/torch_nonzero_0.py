
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def nonzero_inputs():
    list_of_inputs = []

    arr = np.array([1, 0, -2, 0, 3], dtype=np.int32)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array([[0.6, 0.0, 0.0, 0.0],
                    [0.0, 0.4, 0.0, 0.0],
                    [0.0, 0.0, 1.2, 0.0],
                    [0.0, 0.0, 0.0, -0.4]], dtype=np.float32)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array([[[0.0, -1.0, 0.0],
                     [2.5, 0.0, 0.0]],
                    [[0.0, 3.0, 4.0],
                     [0.0, 0.0, -0.7]]], dtype=np.float64)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array([True, False, True, False, False, True], dtype=np.bool_)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array(5, dtype=np.int64)
    out = np.empty((0, 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array(0, dtype=np.int64)
    out = np.empty((0, 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.arange(-12, 12, dtype=np.int64).reshape(1, 2, 3, 4)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array([[0, 255, 0],
                    [1, 0, 2]], dtype=np.uint8)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.zeros((3, 3), dtype=np.float64)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array([], dtype=np.float32)
    out = np.empty((0, 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.empty((2, 0), dtype=np.int32)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array([0+0j, 1+0j, 0-1j, 0+0j], dtype=np.complex64)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array([np.nan, 0.0, -0.0, np.inf, -np.inf], dtype=np.float32)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    arr = np.array([0.0, 1e-4, -1e-4, 0.0, 65500], dtype=np.float16)
    out = np.empty((0, arr.ndim if arr.ndim > 0 else 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": arr, "out": out, "as_tuple": False}))

    return list_of_inputs

generated_inputs["torch.nonzero"] = nonzero_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nonzero'.")


check_valid('torch.nonzero', generated_inputs['torch.nonzero'], lib="torch", suffix=0)
