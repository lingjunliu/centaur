
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def torch_less_equal_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([1.0, 2.0, -3.5], dtype=np.float32)
    other = 0.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 2
    input = np.array([[-1, 0, 1], [2, 3, -4]], dtype=np.int64)
    other = 1.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 3
    input = np.array([[[0.1, -0.2], [1.5, -1.6]], [[-2.0, 3.2], [0.0, -0.0]]], dtype=np.float16)
    other = -1.5
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 4
    input = np.array(5, dtype=np.int32)
    other = 5.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 5
    input = np.array([True, False, True], dtype=bool)
    other = 0.5
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 6
    input = np.array([0, 255], dtype=np.uint8)
    other = 254.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 7
    input = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    other = 0.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 8
    input = np.array([-0.0, 0.0, -1.0, 1.0], dtype=np.float32)
    other = -0.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 9
    input = np.array([], dtype=np.float32)
    other = 1.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 10
    input = np.array([[[10, -20, 30]], [[-40, 50, -60]]], dtype=np.int16)
    other = -100.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 11
    input = np.array([1e308, -1e308], dtype=np.float64)
    other = 0.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 12
    input = np.zeros((2, 0, 3), dtype=np.int64)
    other = 2.0
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 13
    input = np.array([[-1.5, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    other = float('nan')
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    # Input 14
    input = np.array([[1000.0, -1000.0], [np.finfo(np.float32).max, -np.finfo(np.float32).max]], dtype=np.float32)
    other = float('inf')
    out = np.zeros_like(input, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.less_equal_2"] = torch_less_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.less_equal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.less_equal_2'.")


check_valid('torch.less_equal', generated_inputs['torch.less_equal_2'], lib="torch", suffix=2)
