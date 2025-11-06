
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bmm_inputs():
    list_of_inputs = []

    # Input 1: float32, standard shapes
    input_t = torch.randn(2, 3, 4, dtype=torch.float32)
    mat2_t = torch.randn(2, 4, 5, dtype=torch.float32)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.empty((input.shape[0], input.shape[1], mat2.shape[2]), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 2: float64, b=1
    input_t = torch.randn(1, 2, 2, dtype=torch.float64)
    mat2_t = torch.randn(1, 2, 3, dtype=torch.float64)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.zeros((1, 2, 3), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 3: float16
    input_t = torch.randn(4, 1, 6, dtype=torch.float16)
    mat2_t = torch.randn(4, 6, 2, dtype=torch.float16)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.empty((4, 1, 2), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 4: complex64
    input_t = torch.randn(3, 2, 2, dtype=torch.complex64)
    mat2_t = torch.randn(3, 2, 1, dtype=torch.complex64)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.zeros((3, 2, 1), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 5: negative values, float32
    input_t = -torch.rand(5, 1, 3, dtype=torch.float32)
    mat2_t = torch.randn(5, 3, 4, dtype=torch.float32)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.empty((5, 1, 4), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 6: larger batch, square matrices
    input_t = torch.randn(10, 3, 3, dtype=torch.float32)
    mat2_t = torch.randn(10, 3, 3, dtype=torch.float32)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.zeros((10, 3, 3), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 7: float64, non-square
    input_t = torch.randn(7, 2, 5, dtype=torch.float64)
    mat2_t = torch.randn(7, 5, 4, dtype=torch.float64)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.empty((7, 2, 4), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 8: b=1, larger inner dim
    input_t = torch.randn(1, 4, 7, dtype=torch.float32)
    mat2_t = torch.randn(1, 7, 2, dtype=torch.float32)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.zeros((1, 4, 2), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 9: identity input across batch
    input_t = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    mat2_t = torch.randn(2, 3, 3, dtype=torch.float32)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.empty((2, 3, 3), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 10: extreme values, float32
    input_t = (torch.randn(6, 2, 2, dtype=torch.float32) * 1e6)
    mat2_t = (torch.randn(6, 2, 5, dtype=torch.float32) * 1e-6)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.empty((6, 2, 5), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 11: float16, larger dims
    input_t = torch.randn(2, 8, 4, dtype=torch.float16)
    mat2_t = torch.randn(2, 4, 6, dtype=torch.float16)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.zeros((2, 8, 6), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    # Input 12: complex128, minimal shapes
    input_t = torch.randn(2, 1, 1, dtype=torch.complex128)
    mat2_t = torch.randn(2, 1, 1, dtype=torch.complex128)
    input = input_t.numpy()
    mat2 = mat2_t.numpy()
    out = np.zeros((2, 1, 1), dtype=input.dtype)
    list_of_inputs.append(copy.deepcopy({"input": input, "mat2": mat2, "out": out}))

    return list_of_inputs

generated_inputs["torch.bmm"] = bmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bmm'.")


check_valid('torch.bmm', generated_inputs['torch.bmm'], lib="torch", suffix=0)
