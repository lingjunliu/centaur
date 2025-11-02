
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def isposinf_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([-np.inf, np.inf, 0.0], dtype=np.float32)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[np.inf, -np.inf, np.nan],
                      [1.0, 2.5, 3.0]], dtype=np.float64)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([[[np.inf, 0.0, -1.0, 5.5]],
                      [[-np.inf, np.inf, 2.0, -3.0]]], dtype=np.float16)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (scalar)
    input = np.array(np.inf, dtype=np.float64)
    out = np.empty((), dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (empty 1D)
    input = np.array([], dtype=np.float32)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (non-contiguous via transpose)
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    base[0, 0] = np.inf
    base[1, 2] = np.inf
    input = base.T
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (Fortran-ordered)
    input = np.asfortranarray(np.array([[0.0, np.inf, -np.inf],
                                        [np.nan, 7.0, 8.0]], dtype=np.float64))
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (large finite and infinities)
    input = np.array([3.4e38, -3.4e38, np.inf, -np.inf, 1.0], dtype=np.float32)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (zeros, nans, inf)
    input = np.array([[0.0, -0.0, np.nan],
                      [np.inf, 1e-45, -1e-45]], dtype=np.float32)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (3D float64)
    input = np.array([[[1.0, np.inf],
                       [-np.inf, 0.0]],
                      [[np.nan, 5.0],
                       [np.inf, -10.0]]], dtype=np.float64)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (float16 finite only)
    input = np.array([1.0, -2.0, 3.0, 0.0], dtype=np.float16)
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (4D with a single +inf)
    input = np.zeros((1, 2, 1, 3), dtype=np.float32)
    input[0, 1, 0, 2] = np.inf
    out = np.empty(input.shape, dtype=np.bool_)
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.isposinf"] = isposinf_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isposinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isposinf'.")


check_valid('torch.isposinf', generated_inputs['torch.isposinf'], lib="torch", suffix=0)
