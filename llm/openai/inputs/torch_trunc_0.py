
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def trunc_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([3.4742, 0.5466, -0.8008, -0.9079], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 2
    input = np.array([[1.9, -2.1, 3.0],
                      [4.7, -5.5, 6.6]], dtype=np.float64)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 3
    input = np.array([[-1.25, -0.75, 0.75, 1.25]], dtype=np.float16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 4 (0-D scalar)
    input = np.array(3.999, dtype=np.float32)
    out = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 5 (int32)
    input = np.array([-10, -1, 0, 1, 10], dtype=np.int32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 6 (uint8)
    input = np.array([[0, 1, 255],
                      [128, 200, 42]], dtype=np.uint8)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 7 (empty)
    input = np.empty((0,), dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 8 (non-contiguous via slicing)
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    input = base[:, ::-1]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 9 (3D tensor)
    input = np.array(
        [[[1.9, -2.1, 3.3, -4.7],
          [5.5, -6.6, 7.8, -8.9],
          [0.0, -0.0, 9.999, -9.999]],
         [[-1.1, 2.2, -3.3, 4.4],
          [-5.5, 6.6, -7.7, 8.8],
          [10.1, -10.1, 0.1, -0.1]]], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 10 (4D tensor)
    input = np.array([[[[1.2, -1.2],
                        [2.8, -2.8]],
                       [[3.5, -3.5],
                        [4.9, -4.9]]]], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 11 (large/small float64)
    input = np.array([1e20, -1e20, 1e-5, -1e-5], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 12 (int64 large values)
    input = np.array([[1234567890123456789, -1234567890123456789]], dtype=np.int64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.trunc"] = trunc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.trunc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.trunc'.")


check_valid('torch.trunc', generated_inputs['torch.trunc'], lib="torch", suffix=0)
