
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hypot_inputs():
    list_of_inputs = []

    # 1
    input_arr = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    other = np.array([4.0, 3.0, 5.0], dtype=np.float32)
    out = np.empty((3,), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 2
    input_arr = np.array(4.0, dtype=np.float64)
    other = np.array([3.0, 4.0, 5.0], dtype=np.float64)
    out = np.empty((3,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 3
    input_arr = np.array([[1.0, -2.0, 3.0],
                          [4.0, -5.0, 6.0]], dtype=np.float32)
    other = np.array([0.5, -1.5, 2.5], dtype=np.float32)
    out = np.empty((2, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 4
    input_arr = np.arange(2*1*4, dtype=np.float64).reshape(2, 1, 4) - 5.0
    other = (np.arange(1*3*4, dtype=np.float64).reshape(1, 3, 4) / 10.0)
    out = np.empty((2, 3, 4), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 5
    input_arr = np.array([0.0, np.inf, np.nan], dtype=np.float64)
    other = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    out = np.empty((3,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 6
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    input_arr = base[:, ::2]
    other = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    out = np.empty((3, 2), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 7
    input_arr = np.array([[1.0, -1.0, 2.0, -2.0]], dtype=np.float32)
    other = np.array([[0.0],
                      [1.0],
                      [2.0],
                      [3.0],
                      [4.0]], dtype=np.float32)
    out = np.empty((5, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 8
    input_arr = np.array(-3.0, dtype=np.float32)
    other = np.array(4.0, dtype=np.float32)
    out = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 9
    input_arr = np.array([-3.0, -4.0], dtype=np.float32)
    other = np.array([4.0, -3.0], dtype=np.float32)
    out = np.empty((2,), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # 10
    input_arr = np.array([[[1.0], [-2.0], [3.0]], [[-4.0], [5.0], [-6.0]]], dtype=np.float32)
    other = np.array([[[0.0, 1.0, 2.0, 3.0]],
                      [[0.0, -1.0, -2.0, -3.0]]], dtype=np.float32)
    out = np.empty((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.hypot"] = hypot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.hypot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hypot'.")


check_valid('torch.hypot', generated_inputs['torch.hypot'], lib="torch", suffix=0)
