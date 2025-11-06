
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def absolute_inputs():
    list_of_inputs = []

    input = np.array([-1.0, 2.5, -3.3, 0.0], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[-1.0, 2.0], [3.5, -4.5]], dtype=np.float64)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[[-1, 2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array(-123.456, dtype=np.float32)
    out = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[[[-1, -128], [100, -100]]]], dtype=np.int8)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([1+2j, -3-4j, 0+0j], dtype=np.complex64)
    out = np.empty(input.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[1+1j, -2+2j], [3-3j, -4-4j]], dtype=np.complex128)
    out = np.zeros(input.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[0, 255], [128, 1]], dtype=np.uint8)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([-9223372036854775808, -1, 0, 1, 9223372036854775807], dtype=np.int64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[-1.5, 2.25, -0.0], [3.125, -4.5, 5.0], [-6.75, 7.0, -8.875]], dtype=np.float16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = np.arange(-12, 0, dtype=np.float32).reshape(3, 4)
    input = base[:, ::2]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    mat = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int16)
    input = mat.T
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.absolute"] = absolute_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.absolute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.absolute'.")


check_valid('torch.absolute', generated_inputs['torch.absolute'], lib="torch", suffix=0)
