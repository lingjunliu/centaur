
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_det_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = np.array(0.0, dtype=np.float32)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of matrices
    A = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    out = np.array([0.0, 0.0], dtype=np.float64)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger matrix
    A = np.random.rand(5, 5).astype(np.float32)
    out = np.array(0.0, dtype=np.float32)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrix with negative values
    A = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    out = np.array(0.0, dtype=np.float64)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D batch, different size
    A = np.random.rand(2, 3, 3).astype(np.float32)
    out = np.array([0.0, 0.0], dtype=np.float32)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Complex matrix
    A = np.array([[1+1j, 2-1j], [3+0j, 4+2j]], dtype=np.complex64)
    out = np.array(0+0j, dtype=np.complex64)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch of complex matrices
    A = np.array([[[1+1j, 2-1j], [3+0j, 4+2j]], [[5-1j, 6+1j], [7+2j, 8-0j]]], dtype=np.complex128)
    out = np.array([0+0j, 0+0j], dtype=np.complex128)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Identity matrix
    A = np.eye(3, dtype=np.float32)
    out = np.array(0.0, dtype=np.float32)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Larger complex matrix
    A = (np.random.rand(4, 4) + 1j * np.random.rand(4, 4)).astype(np.complex64)
    out = np.array(0+0j, dtype=np.complex64)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched Identity Matrices
    A = np.stack([np.eye(2, dtype=np.float64) for _ in range(3)])
    out = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.det"] = linalg_det_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.det' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.det'.")

check_valid('torch.linalg.det', generated_inputs['torch.linalg.det'], lib="torch", suffix=0)
