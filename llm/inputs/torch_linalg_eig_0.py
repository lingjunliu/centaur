
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_linalg_eig_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 real matrix
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    eigenvalues = np.zeros(2, dtype=np.complex128)
    eigenvectors = np.zeros((2, 2), dtype=np.complex128)
    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2x2 real matrices
    A = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    eigenvalues = np.zeros((2, ), dtype=np.complex128)
    eigenvectors = np.zeros((2, 2, 2), dtype=np.complex128)

    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3x3 real matrix
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    eigenvalues = np.zeros(3, dtype=np.complex128)
    eigenvectors = np.zeros((3, 3), dtype=np.complex128)
    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex 2x2 matrix
    A = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], dtype=np.complex128)
    eigenvalues = np.zeros(2, dtype=np.complex128)
    eigenvectors = np.zeros((2, 2), dtype=np.complex128)
    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch of Complex 2x2 matrices
    A = np.array([[[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], [[5.0, 6.0 + 2j], [7.0 - 1j, 8.0]]], dtype=np.complex128)
    eigenvalues = np.zeros((2,), dtype=np.complex128)
    eigenvectors = np.zeros((2, 2, 2), dtype=np.complex128)
    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float32 2x2
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    eigenvalues = np.zeros(2, dtype=np.complex64)
    eigenvectors = np.zeros((2, 2), dtype=np.complex64)
    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Complex Float32 2x2
    A = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]], dtype=np.complex64)
    eigenvalues = np.zeros(2, dtype=np.complex64)
    eigenvectors = np.zeros((2, 2), dtype=np.complex64)
    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4x4 real matrix
    A = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], dtype=np.float64)
    eigenvalues = np.zeros(4, dtype=np.complex128)
    eigenvectors = np.zeros((4, 4), dtype=np.complex128)
    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Diagonal matrix
    A = np.diag([1.0, 2.0, 3.0])
    eigenvalues = np.zeros(3, dtype=np.complex128)
    eigenvectors = np.zeros((3, 3), dtype=np.complex128)
    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Complex matrix with zero imaginary parts
    A = np.array([[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]], dtype=np.complex128)
    eigenvalues = np.zeros(2, dtype=np.complex128)
    eigenvectors = np.zeros((2, 2), dtype=np.complex128)
    out = (eigenvalues, eigenvectors)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.eig"] = torch_linalg_eig_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.eig' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eig'.")

check_valid('torch.linalg.eig', generated_inputs['torch.linalg.eig'], lib="torch", suffix=0)
