
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_slogdet_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 matrix
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = (np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 matrix with negative values
    A = np.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [7.0, 8.0, -9.0]], dtype=np.float64)
    out = (np.array(0.0, dtype=np.float64), np.array(0.0, dtype=np.float64))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of 2 matrices (2x2)
    A = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    out = (np.array([0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrix with determinant close to zero
    A = np.array([[1e-8, 1.0], [1.0, 1.0]], dtype=np.float32)
    out = (np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex matrix
    A = np.array([[1.0 + 1j, 2.0 - 1j], [3.0 + 0j, 4.0 - 2j]], dtype=np.complex64)
    out = (np.array(0.0 + 0.0j, dtype=np.complex64), np.array(0.0, dtype=np.float32))

    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger matrix (4x4)
    A = np.random.rand(4, 4).astype(np.float64)
    out = (np.array(0.0, dtype=np.float64), np.array(0.0, dtype=np.float64))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch of complex matrices
    A = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex128)
    out = (np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128), np.array([0.0, 0.0], dtype=np.float64))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix with zero determinant
    A = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    out = (np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1x1 Matrix
    A = np.array([[5.0]], dtype=np.float32)
    out = (np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batch of 1x1 matrices
    A = np.array([[[2.0]], [[3.0]]], dtype=np.float32)
    out = (np.array([0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Complex 1x1 Matrix
    A = np.array([[5.0 + 2j]], dtype=np.complex64)
    out = (np.array(0.0 + 0.0j, dtype=np.complex64), np.array(0.0, dtype=np.float32))
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.slogdet"] = linalg_slogdet_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.slogdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.slogdet'.")

check_valid('torch.linalg.slogdet', generated_inputs['torch.linalg.slogdet'], lib="torch", suffix=0)
