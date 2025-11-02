
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def torch_multiply_2_inputs():
    list_of_inputs = []

    # 1
    input = np.array([1.0, -2.5, 3.0], dtype=np.float32)
    other = 2.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 2
    input = np.array(5.0, dtype=np.float64)
    other = -1.5
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 3
    input = np.arange(6, dtype=np.int32).reshape(2, 3)
    other = np.float32(0.5)
    out = np.empty(input.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 4
    input = (np.ones((2, 2, 3), dtype=np.float16) * -3)
    other = 4.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 5
    input = np.array([], dtype=np.float32)
    other = 3.14
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 6
    input = np.empty((2, 0, 3), dtype=np.float64)
    other = -2.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 7
    input = np.array([10, 20, 30, 255], dtype=np.uint8)
    other = 10.0
    out = np.empty(input.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 8
    input = np.array([1+2j, -3+0.5j, 0-1j], dtype=np.complex64)
    other = 1.25
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 9
    input = np.asfortranarray(np.arange(9, dtype=np.float64).reshape(3, 3))
    other = np.float64(0.0)
    out = np.empty_like(input, order='F')
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 10
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    input = base[:, ::2]
    other = -0.75
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 11
    input = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float64)
    other = 2.5
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # 12
    input = np.array([10**12, -10**12, 42], dtype=np.int64)
    other = 1e-3
    out = np.empty(input.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.multiply_2"] = torch_multiply_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.multiply_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.multiply_2'.")


check_valid('torch.multiply', generated_inputs['torch.multiply_2'], lib="torch", suffix=2)
