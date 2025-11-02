
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sqrt_inplace_inputs():
    list_of_inputs = []

    input = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([[-1.0, 0.0, 1.0],
                      [2.0, -3.0, 16.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([[[0.25, 1.0, 4.0],
                       [9.0, 16.0, 25.0]],
                      [[36.0, 49.0, 64.0],
                       [81.0, 100.0, 121.0]]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array(-4.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.arange(12, dtype=np.float32).reshape(3, 4).T
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([np.inf, -np.inf, np.nan, 0.0, 100.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([1+0j, -1+0j, 3-4j, 0+0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([[-1+1j, 2-3j],
                      [4+0j, 0+5j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (np.arange(12, dtype=np.float32).reshape(2, 1, 3, 1, 2) / 3.0) + 0.1
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.array([1e-45, 1e-38, 1e-20, 1e-10, 1e-5, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.sqrt_"] = sqrt_inplace_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sqrt_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sqrt_'.")


check_valid('torch.sqrt_', generated_inputs['torch.sqrt_'], lib="torch", suffix=0)
