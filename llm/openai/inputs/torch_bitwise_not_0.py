
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def bitwise_not_inputs():
    list_of_inputs = []

    input = np.array([True, False, True, True, False], dtype=np.bool_)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[-128, -1, 0, 1, 127],
                      [5, -5, 10, -10, 42]], dtype=np.int8)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[[ -1,   2,  -3],
                       [  4,  -5,   6]],
                      [[ -7,   8,  -9],
                       [ 10, -11,  12]]], dtype=np.int16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array(42, dtype=np.int32)
    out = np.empty((), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([np.iinfo(np.int64).min, -1234567890123456789, -2, -1, 0, 1, 2, np.iinfo(np.int64).max], dtype=np.int64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[0, 1, 2, 255],
                      [128, 64, 32, 16]], dtype=np.uint8)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([], dtype=np.int32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = np.arange(24, dtype=np.int32).reshape(4, 6)
    input = base[:, ::2]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base_bool = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    input = base_bool.T
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.arange(2*3*4*5, dtype=np.int64).reshape(2, 3, 4, 5)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.array([[0, 255, 1], [2, 3, 4], [5, 6, 7]], dtype=np.uint8)[::2]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = np.zeros((2, 0, 3), dtype=np.int16)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.bitwise_not"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bitwise_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_not'.")


check_valid('torch.bitwise_not', generated_inputs['torch.bitwise_not'], lib="torch", suffix=0)
