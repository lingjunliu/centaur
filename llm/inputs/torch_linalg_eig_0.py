
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_eig_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 real matrix
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    out = None
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple 2x2 complex matrix
    A = np.array([[1.0+1j, 2.0], [3.0, 4.0-1j]], dtype=np.complex128)
    out = None
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of 2 2x2 real matrices
    A = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    out = None
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3x3 matrix with float32
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    out = None
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3x3 matrix with complex64
    A = np.array([[1.0+1j, 2.0, 3.0], [4.0, 5.0-1j, 6.0], [7.0, 8.0, 9.0+2j]], dtype=np.complex64)
    out = None
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch of 2 3x3 real matrices
    A = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]], dtype=np.float64)
    out = None
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values
    A = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    out = None
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero matrix
    A = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    out = None
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Identity matrix
    A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    out = None
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger matrix (4x4)
    A = np.random.rand(4, 4).astype(np.float64)
    out = None
    input_dict = {"A": A, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Example with 'out' specified as a tuple of numpy arrays
    # Initialize out with correct shape and dtype
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    eigenvalues_out = np.zeros(A.shape[0], dtype=np.complex128)
    eigenvectors_out = np.eye(A.shape[0], dtype=np.complex128) #Initialize with identity matrix
    out = (eigenvalues_out, eigenvectors_out)
    input_dict = {"A": A, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.eig"] = linalg_eig_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.eig' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eig'.")

check_valid('torch.linalg.eig', generated_inputs['torch.linalg.eig'], lib="torch", suffix=0)
