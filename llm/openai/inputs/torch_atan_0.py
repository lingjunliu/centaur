
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def atan_inputs():
    list_of_inputs = []

    input = np.array([-1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[0.1, -0.2, 0.3], [1.5, -2.5, 3.5]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array(0.5, dtype=np.float16)
    out = np.empty((), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], [[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6]]], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([1+1j, -2+0.5j, -0.5-3j, 0+0j], dtype=np.complex64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[1-2j, -3+4j], [5+0j, -6-7j]], dtype=np.complex128)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([np.inf, -np.inf, np.nan, 0.0, -1.0], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0], dtype=np.float16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.ones((1, 2, 2, 2), dtype=np.float32) * 3.14159
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([-3, -1, 0, 1, 3], dtype=np.int32)
    out = np.empty_like(input, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[1e-30, -1e-20, 1e-10], [-1e10, 1e20, -1e30]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.atan"] = atan_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.atan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atan'.")


check_valid('torch.atan', generated_inputs['torch.atan'], lib="torch", suffix=0)
