
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def mm_inputs():
    list_of_inputs = []

    input_arr = torch.randn(2, 3, dtype=torch.float32).numpy()
    mat2_arr = torch.randn(3, 4, dtype=torch.float32).numpy()
    out_arr = np.zeros((2, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = torch.randn(3, 3, dtype=torch.float64).numpy()
    mat2_arr = torch.randn(3, 2, dtype=torch.float64).numpy()
    out_arr = np.zeros((3, 2), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = np.array([[1], [-2], [3], [0]], dtype=np.int32)
    mat2_arr = np.array([[5, -1, 2, 7, 0]], dtype=np.int32)
    out_arr = np.zeros((4, 5), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = np.array([[7]], dtype=np.int64)
    mat2_arr = np.array([[3]], dtype=np.int64)
    out_arr = np.zeros((1, 1), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = (np.random.randn(5, 2)).astype(np.float16)
    mat2_arr = (np.random.randn(2, 3)).astype(np.float16)
    out_arr = np.zeros((5, 3), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    mat2_arr = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    out_arr = np.zeros((2, 2), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = (np.random.randn(3, 1) + 1j * np.random.randn(3, 1)).astype(np.complex128)
    mat2_arr = (np.random.randn(1, 3) + 1j * np.random.randn(1, 3)).astype(np.complex128)
    out_arr = np.zeros((3, 3), dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    base = np.arange(12, dtype=np.float32).reshape(4, 3)
    input_arr = base.T
    mat2_arr = np.random.randn(4, 2).astype(np.float32)
    out_arr = np.zeros((3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = np.asfortranarray(np.random.randn(2, 3).astype(np.float64))
    mat2_arr = np.asfortranarray(np.random.randn(3, 2).astype(np.float64))
    out_arr = np.zeros((2, 2), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = np.zeros((0, 3), dtype=np.float32)
    mat2_arr = np.zeros((3, 2), dtype=np.float32)
    out_arr = np.zeros((0, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = np.array([[2, -3, 4, -5, 6],
                          [-1, 0, 1, -2, 3]], dtype=np.int16)
    mat2_arr = np.array([[1, 2],
                         [-1, 0],
                         [3, -3],
                         [2, 1],
                         [-2, 4]], dtype=np.int16)
    out_arr = np.zeros((2, 2), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    input_arr = np.random.randn(1, 6).astype(np.float32)
    mat2_arr = np.random.randn(6, 1).astype(np.float32)
    out_arr = np.zeros((1, 1), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "mat2": mat2_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.mm"] = mm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.mm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mm'.")


check_valid('torch.mm', generated_inputs['torch.mm'], lib="torch", suffix=0)
