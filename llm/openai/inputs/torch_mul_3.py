
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def mul_inputs():
    list_of_inputs = []

    input = np.array([0.2, -0.4, 2.6], dtype=np.float32)
    other = 100
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = np.array([[1, -2, 3], [4, 0, -6]], dtype=np.int64)
    other = -3
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 5.0
    other = -2
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = np.array(5.5, dtype=np.float64)
    other = 7
    out = np.empty((), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = np.empty((0, 3), dtype=np.float32)
    other = 5
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = np.array([2**30, -(2**30) + 1], dtype=np.int32)
    other = 2
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = np.array([[1+2j, -3+0.5j], [-1j, 2+3j]], dtype=np.complex64)
    other = 4
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = (np.arange(60, dtype=np.float16).reshape(3, 4, 5) / np.float16(10)) - np.float16(3)
    other = 0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = np.array([[[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]],
                      [[[9, -10], [11, -12]], [[13, -14], [15, -16]]]], dtype=np.int8)
    other = np.int8(-1)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    input = base[:, ::2] - 10.0
    other = np.int32(5)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = np.array([[0, 127], [200, 255]], dtype=np.uint8)
    other = 3
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    rng = np.random.default_rng(0)
    input = (rng.standard_normal((1, 2, 1, 3, 4)) + 1j * rng.standard_normal((1, 2, 1, 3, 4))).astype(np.complex128)
    other = np.int64(2)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.mul_3"] = mul_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.mul_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mul_3'.")


check_valid('torch.mul', generated_inputs['torch.mul_3'], lib="torch", suffix=3)
