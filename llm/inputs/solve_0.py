
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_linalg_solve_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    A = np.random.rand(3, 3).astype(np.float32)
    B = np.random.rand(3, 1).astype(np.float32)
    input_dict = {"A": A, "B": B, "left": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Double tensors with multiple right-hand sides, left=True
    A = np.random.rand(2, 2).astype(np.float64)
    B = np.random.rand(2, 3).astype(np.float64)
    input_dict = {"A": A, "B": B, "left": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Complex tensors
    A = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(np.complex64)
    B = (np.random.rand(3, 1) + 1j * np.random.rand(3, 1)).astype(np.complex64)
    input_dict = {"A": A, "B": B, "left": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Batched matrices
    A = np.random.rand(2, 3, 3).astype(np.float32)
    B = np.random.rand(2, 3, 2).astype(np.float32)
    input_dict = {"A": A, "B": B, "left": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Single vector B
    A = np.random.rand(3, 3).astype(np.float32)
    B = np.random.rand(3).astype(np.float32)
    input_dict = {"A": A, "B": B, "left": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.solve"] = torch_linalg_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.solve'.")

check_valid('torch.linalg.solve', generated_inputs['torch.linalg.solve'], lib="torch")
