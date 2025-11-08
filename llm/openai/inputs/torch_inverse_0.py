
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def inverse_inputs():
    list_of_inputs = []

    # 1: 2x2 float32
    input = np.array([[2.0, 1.0],
                      [3.0, 2.0]], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 2: 3x3 float64, upper triangular with non-zero diagonal
    input = np.array([[3.0, 0.0, 1.0],
                      [0.0, -2.0, 4.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 3: 1x1 float32
    input = np.array([[5.0]], dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 4: batched 2x2 float64 (batch=2)
    input = np.array([[[1.0, 2.0],
                       [3.0, 4.0]],
                      [[5.0, 6.0],
                       [7.0, 10.0]]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 5: batched 3x3 float32 SPD-like
    A1 = np.array([[2.0, 0.1, 0.0],
                   [0.1, 2.0, 0.2],
                   [0.0, 0.2, 2.0]], dtype=np.float32)
    A2 = np.array([[1.5, -0.2, 0.3],
                   [-0.2, 1.8, 0.1],
                   [0.3, 0.1, 1.6]], dtype=np.float32)
    A3 = np.array([[3.0, 0.0, -0.1],
                   [0.0, 3.0, 0.2],
                   [-0.1, 0.2, 3.0]], dtype=np.float32)
    input = np.stack([A1, A2, A3], axis=0)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 6: 2x2 complex64 diagonal
    input = np.array([[1+2j, 0+0j],
                      [0+0j, 3-1j]], dtype=np.complex64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 7: 3x3 complex128 upper triangular
    input = np.array([[1+0.5j, 2-1j, 0+0j],
                      [0+0j, -2+2j, 0+1j],
                      [0+0j, 0+0j, 3-0.5j]], dtype=np.complex128)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 8: 4x4 float32 permutation (orthogonal)
    P = np.eye(4, dtype=np.float32)
    input = P[[2, 0, 3, 1], :]
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 9: 2x2 float64 with negatives and fractions
    input = np.array([[-1.5, 0.5],
                      [0.2, 2.0]], dtype=np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 10: batched 4x4 float64 lower triangular with non-zero diagonal (batch=2)
    L1 = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.5, 2.0, 0.0, 0.0],
                   [-0.3, 0.7, 3.0, 0.0],
                   [0.2, -0.1, 0.4, 4.0]], dtype=np.float64)
    L2 = np.array([[2.0, 0.0, 0.0, 0.0],
                   [1.0, 3.0, 0.0, 0.0],
                   [0.5, -0.2, 4.0, 0.0],
                   [-0.1, 0.3, 0.6, 5.0]], dtype=np.float64)
    input = np.stack([L1, L2], axis=0)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 11: empty batch (0, 2, 2) float32
    input = np.empty((0, 2, 2), dtype=np.float32)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 12: 5x5 float64 diagonal
    input = np.diag(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 13: multi-batch (2, 3, 2, 2) float32, diagonal matrices
    base = np.array([[2.0, 0.0],
                     [0.0, 3.0]], dtype=np.float32)
    input = np.tile(base.reshape(1, 1, 2, 2), (2, 3, 1, 1))
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # 14: batched 2x2 complex64 (batch=2)
    M1 = np.array([[1+1j, 0+0j],
                   [0+0j, 2-1j]], dtype=np.complex64)
    M2 = np.array([[-1+2j, 0+0j],
                   [0+0j, 0.5+0.5j]], dtype=np.complex64)
    input = np.stack([M1, M2], axis=0)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.inverse"] = inverse_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.inverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.inverse'.")


check_valid('torch.inverse', generated_inputs['torch.inverse'], lib="torch", suffix=0)
