
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def qr_inputs():
    list_of_inputs = []

    # Input 1
    A = np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float64)
    mode = 'reduced'
    out = (np.array([[1.0]], dtype=np.float64), np.array([[1.0]], dtype=np.float64))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    A = np.random.rand(5, 3).astype(np.float64)
    mode = 'complete'
    out = (np.random.rand(1,1).astype(np.float64), np.random.rand(1,1).astype(np.float64))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    A = np.random.rand(3, 5).astype(np.float64)
    mode = 'r'
    out = (np.random.rand(1,1).astype(np.float64), np.random.rand(1,1).astype(np.float64))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of matrices
    A = np.random.rand(2, 4, 4).astype(np.float64)
    mode = 'reduced'
    out = (np.random.rand(1,1).astype(np.float64), np.random.rand(1,1).astype(np.float64))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex numbers
    A = (np.random.rand(3, 3) + 1j*np.random.rand(3, 3)).astype(np.complex128)
    mode = 'reduced'
    out = (np.random.rand(1,1).astype(np.complex128), np.random.rand(1,1).astype(np.complex128))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different size matrix
    A = np.random.rand(2, 5).astype(np.float64)
    mode = 'complete'
    out = (np.random.rand(1,1).astype(np.float64), np.random.rand(1,1).astype(np.float64))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Taller matrix
    A = np.random.rand(5, 2).astype(np.float64)
    mode = 'reduced'
    out = (np.random.rand(1,1).astype(np.float64), np.random.rand(1,1).astype(np.float64))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D batch
    A = np.random.rand(4, 3, 2).astype(np.float64)
    mode = 'r'
    out = (np.random.rand(1,1).astype(np.float64), np.random.rand(1,1).astype(np.float64))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero matrix
    A = np.zeros((3, 3)).astype(np.float64)
    mode = 'reduced'
    out = (np.random.rand(1,1).astype(np.float64), np.random.rand(1,1).astype(np.float64))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Identity matrix
    A = np.eye(3).astype(np.float64)
    mode = 'complete'
    out = (np.random.rand(1,1).astype(np.float64), np.random.rand(1,1).astype(np.float64))
    input_dict = {"A": A, "mode": mode, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.qr"] = qr_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.qr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.qr'.")

check_valid('torch.linalg.qr', generated_inputs['torch.linalg.qr'], lib="torch", suffix=0)
