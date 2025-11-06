
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def matmul_inputs():
    list_of_inputs = []
    rng = np.random.default_rng(42)

    # 1) 2D x 2D, float32
    input_arr = rng.normal(size=(3, 4)).astype(np.float32)
    other_arr = rng.normal(size=(4, 2)).astype(np.float32)
    out_arr = np.zeros((3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 2) 2D x 1D, float64
    input_arr = rng.normal(size=(3, 4)).astype(np.float64)
    other_arr = rng.normal(size=(4,)).astype(np.float64)
    out_arr = np.zeros((3,), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 3) 1D x 2D, float32
    input_arr = rng.normal(size=(5,)).astype(np.float32)
    other_arr = rng.normal(size=(5, 6)).astype(np.float32)
    out_arr = np.zeros((6,), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 4) Batched matrix x batched matrix, float32
    input_arr = rng.normal(size=(10, 3, 4)).astype(np.float32)
    other_arr = rng.normal(size=(10, 4, 5)).astype(np.float32)
    out_arr = np.zeros((10, 3, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 5) Batched matrix x broadcasted matrix, float32
    input_arr = rng.normal(size=(10, 3, 4)).astype(np.float32)
    other_arr = rng.normal(size=(4, 5)).astype(np.float32)
    out_arr = np.zeros((10, 3, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 6) Higher-dim batched with broadcasting, float32
    input_arr = rng.normal(size=(2, 3, 4, 6)).astype(np.float32)
    other_arr = rng.normal(size=(1, 6, 7)).astype(np.float32)
    out_arr = np.zeros((2, 3, 4, 7), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 7) 2D x 2D, int32
    input_arr = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    other_arr = np.array([[7, 8, -1, 2], [0, -3, 4, 5], [6, -7, 8, 9]], dtype=np.int32)
    out_arr = np.zeros((2, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 8) 2D x 2D, complex64
    input_arr = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))).astype(np.complex64)
    other_arr = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))).astype(np.complex64)
    out_arr = np.zeros((2, 2), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 9) Batched matrix x vector, float16
    input_arr = rng.normal(size=(5, 2, 3, 4)).astype(np.float16)
    other_arr = rng.normal(size=(4,)).astype(np.float16)
    out_arr = np.zeros((5, 2, 3), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 10) Vector x batched matrix, float32
    input_arr = rng.normal(size=(4,)).astype(np.float32)
    other_arr = rng.normal(size=(6, 4, 3)).astype(np.float32)
    out_arr = np.zeros((6, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 11) Non-contiguous input via transpose, float64
    base = np.arange(24, dtype=np.float64).reshape(6, 4)
    input_arr = base.T  # shape (4, 6), non-contiguous
    other_arr = np.arange(18, dtype=np.float64).reshape(6, 3)
    out_arr = np.zeros((4, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    # 12) Batched broadcasting with singleton dims, float32
    input_arr = rng.normal(size=(3, 1, 2, 2)).astype(np.float32)
    other_arr = rng.normal(size=(1, 2, 2)).astype(np.float32)
    out_arr = np.zeros((3, 1, 2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "other": other_arr, "out": out_arr}))

    return list_of_inputs

generated_inputs["torch.matmul"] = matmul_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.matmul'.")


check_valid('torch.matmul', generated_inputs['torch.matmul'], lib="torch", suffix=0)
