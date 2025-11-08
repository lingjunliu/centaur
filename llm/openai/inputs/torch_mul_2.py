
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def mul_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([1.0, -2.5, 3.25], dtype=np.float32)
    other = 2.5
    out = np.empty_like(input_arr, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 2
    input_arr = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    other = -1.0
    out = np.empty(input_arr.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 3
    input_arr = np.arange(8, dtype=np.int64).reshape(4, 1, 2)
    other = 0.0
    out = np.empty(input_arr.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 4
    input_arr = np.array([[np.pi, -np.e, 1.0]], dtype=np.float64)
    other = -3.5
    out = np.empty(input_arr.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 5
    input_arr = np.array([[True, False, True], [False, True, False]], dtype=bool)
    other = 10.0
    out = np.empty(input_arr.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 6
    input_arr = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    other = 1.5
    out = np.empty(input_arr.shape, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 7
    input_arr = (np.random.randn(2, 3, 1) + 1j * np.random.randn(2, 3, 1)).astype(np.complex128)
    other = -2.0
    out = np.empty(input_arr.shape, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 8
    input_arr = np.array(3.14, dtype=np.float32)
    other = 3.0
    out = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 9
    input_arr = np.array([], dtype=np.float32)
    other = 7.0
    out = np.empty_like(input_arr, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 10
    input_arr = np.empty((2, 0, 3), dtype=np.float32)
    other = np.float32(0.5)
    out = np.empty(input_arr.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 11
    input_arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    other = np.float64(1e-3)
    out = np.empty(input_arr.shape, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    # Input 12
    input_arr = np.array([-10, -5, 0, 5, 10, 20], dtype=np.int16)
    other = np.float32(-0.25)
    out = np.empty(input_arr.shape, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.mul_2"] = mul_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.mul_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mul_2'.")


check_valid('torch.mul', generated_inputs['torch.mul_2'], lib="torch", suffix=2)
