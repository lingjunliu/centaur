
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fix_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 with negatives, positives, -0.0
    input = np.array([-2.7, -0.0, 0.0, 1.5, 2.999, -3.001], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 2: 2D float64 with NaN and Inf
    input = np.array([[-1.9, 2.2, np.nan],
                      [np.inf, -np.inf, 0.4999]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 3: 3D float16
    input = np.array([[[1.1, -1.9, 2.5],
                       [3.9, -4.4, 0.0]],
                      [[-0.1, 0.1, 5.7],
                       [7.8, -8.9, 9.99]]], dtype=np.float16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 4: scalar float32 (0-D)
    input = np.array(-5.9, dtype=np.float32)
    out = np.empty((), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 5: 1D int64
    input = np.array([-10, -1, 0, 1, 10, 1234567890123], dtype=np.int64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 6: 2D uint8
    input = np.array([[0, 1, 2],
                      [253, 254, 255]], dtype=np.uint8)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 7: 4D float32
    input = np.array([[[[ 0.99, -0.99,  1.01],
                        [ 2.49,  2.50,  2.51]]],
                      [[[ -3.49, -3.50, -3.51],
                        [ 10.75, -10.75, 0.25]]]], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 8: non-contiguous float64 slice
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    input = base[:, ::2] + 0.75
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 9: empty array float32
    input = np.array([], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 10: 3D int16
    input = np.array([[[-32768], [-123], [0]],
                      [[123], [32767], [-1]]], dtype=np.int16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 11: reversed stride 1D float32
    input = (np.linspace(-5, 5, num=11, dtype=np.float32)[::-1] + 0.3).astype(np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 12: large magnitude float64
    input = np.array([1e20 + 0.9, -1e-20 - 0.9, -1e20 + 0.1, 3.9999999999], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.fix"] = fix_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fix'.")


check_valid('torch.fix', generated_inputs['torch.fix'], lib="torch", suffix=0)
