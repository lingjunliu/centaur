
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def eq_2_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    other = 2.0
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 2
    input_arr = torch.tensor([[1.0, float('nan')], [float('inf'), float('-inf')]], dtype=torch.float32).numpy()
    other = np.float32(float('inf'))
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 3
    input_arr = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    other = 1.0
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 4
    input_arr = torch.tensor([-3, 0, 3, 0], dtype=torch.int8).numpy()
    other = -0.0
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 5
    input_arr = torch.tensor([[[0.5, -0.5, 0.0]], [[1.5, 0.5, 2.5]]], dtype=torch.float64).numpy()
    other = 0.5
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 6
    input_arr = torch.tensor([1+0j, 2+3j, 0+0j], dtype=torch.complex64).numpy()
    other = 1.0
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 7
    input_arr = torch.tensor(5, dtype=torch.int32).numpy()
    other = 5.0
    out = np.empty((), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 8
    input_arr = (torch.ones((1, 2, 3, 1), dtype=torch.float32) * -1.0).numpy()
    other = -1.0
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 9
    input_arr = torch.tensor([0, 255], dtype=torch.uint8).numpy()
    other = 255.0
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 10
    input_arr = torch.tensor([[1.0, 1.0, 1.0],
                              [1.0, 0.0, -1.0],
                              [float('nan'), float('inf'), float('-inf')]], dtype=torch.float16).numpy()
    other = np.float16(1.0)
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 11
    input_arr = torch.empty((2, 0, 4), dtype=torch.int64).numpy()
    other = 0.0
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 12
    input_arr = torch.arange(-4, 4, dtype=torch.float32).reshape(2, 4).numpy()
    other = -2.5
    out = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.eq_2"] = eq_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.eq_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.eq_2'.")


check_valid('torch.eq', generated_inputs['torch.eq_2'], lib="torch", suffix=2)
