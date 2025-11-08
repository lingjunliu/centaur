
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def mv_inputs():
    list_of_inputs = []

    # Input 1: float32, typical case
    input_arr = np.array([[1.0, -2.0, 3.0],
                          [4.5, 0.0, -1.5]], dtype=np.float32)
    vec = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    out = np.zeros(2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 2: float64 with negatives, (3x1) @ (1,)
    input_arr = np.array([[-1.0],
                          [2.5],
                          [-3.5]], dtype=np.float64)
    vec = np.array([4.0], dtype=np.float64)
    out = np.empty(3, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 3: int32
    input_arr = np.array([[1, 2],
                          [-3, 4],
                          [5, -6],
                          [7, 8]], dtype=np.int32)
    vec = np.array([9, -10], dtype=np.int32)
    out = np.zeros(4, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 4: int64 single element
    input_arr = np.array([[1234567890123456789]], dtype=np.int64)
    vec = np.array([-2], dtype=np.int64)
    out = np.empty(1, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 5: complex64 square matrix
    input_arr = np.array([[1+2j, -3+0.5j, 2-1j],
                          [0+0j, 4-4j, -5+2j],
                          [7+3j, 8+0j, -9-9j]], dtype=np.complex64)
    vec = np.array([2-1j, -1+2j, 0.5+0.5j], dtype=np.complex64)
    out = np.zeros(3, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 6: complex128 rectangular
    input_arr = np.array([[1-1j, 2+2j, 3-3j, 4+4j],
                          [-1+1j, -2-2j, -3+3j, -4-4j]], dtype=np.complex128)
    vec = np.array([0.1+0.2j, -0.3+0.4j, 0.5-0.6j, -0.7-0.8j], dtype=np.complex128)
    out = np.empty(2, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 7: empty output dimension n=0
    input_arr = np.zeros((0, 3), dtype=np.float32)
    vec = np.array([1.0, -1.0, 2.0], dtype=np.float32)
    out = np.zeros((0,), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 8: inner dimension m=0
    input_arr = np.zeros((5, 0), dtype=np.float64)
    vec = np.zeros((0,), dtype=np.float64)
    out = np.zeros((5,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 9: Fortran-ordered input
    base = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]], dtype=np.float32)
    input_arr = np.asfortranarray(base)
    vec = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 10: non-contiguous view via slicing
    B = np.arange(20, dtype=np.float64).reshape(4, 5)
    input_arr = B[:, ::2]  # shape (4,3)
    vec = np.array([0.5, -1.5, 2.0], dtype=np.float64)
    out = np.empty(4, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 11: int16
    input_arr = np.array([[300, -200, 100],
                          [-100, 0, 50]], dtype=np.int16)
    vec = np.array([1, -2, 3], dtype=np.int16)
    out = np.zeros(2, dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 12: float32 with out as a slice of a larger buffer
    input_arr = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(4, 6)
    vec = np.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32)
    out_buffer = np.zeros(10, dtype=np.float32)
    out = out_buffer[3:7]
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    # Input 13: small magnitudes float64
    input_arr = np.array([[1e-8, -1e-9],
                          [1e-12, 1e-6]], dtype=np.float64)
    vec = np.array([1e9, -1e3], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "vec": vec, "out": out}))

    return list_of_inputs

generated_inputs["torch.mv"] = mv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.mv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mv'.")


check_valid('torch.mv', generated_inputs['torch.mv'], lib="torch", suffix=0)
