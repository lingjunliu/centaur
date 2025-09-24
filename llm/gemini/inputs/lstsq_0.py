
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lstsq_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    A = np.random.randn(3, 2).astype(np.float32)
    B = np.random.randn(3, 1).astype(np.float32)
    input_dict = {"A": A, "B": B, "rcond": 1e-15, "driver": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different shapes, double precision
    A = np.random.randn(5, 3).astype(np.float64)
    B = np.random.randn(5, 2).astype(np.float64)
    input_dict = {"A": A, "B": B, "rcond": 1e-10, "driver": 'gelsd'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Batch dimensions, complex numbers
    A = np.random.randn(2, 4, 3).astype(np.complex64)
    B = np.random.randn(2, 4, 2).astype(np.complex64)
    input_dict = {"A": A, "B": B, "rcond": 1e-8, "driver": 'gelsy'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Rectangular matrix, negative values
    A = np.random.randn(2, 5).astype(np.float32)
    B = np.random.randn(2, 3).astype(np.float32)
    A *= -1
    B *= -1
    input_dict = {"A": A, "B": B, "rcond": None, "driver": 'gelss'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Small matrices
    A = np.random.randn(1, 1).astype(np.float64)
    B = np.random.randn(1, 1).astype(np.float64)
    input_dict = {"A": A, "B": B, "rcond": 1e-12, "driver": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.lstsq"] = lstsq_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.lstsq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.lstsq'.")

check_valid('torch.linalg.lstsq', generated_inputs['torch.linalg.lstsq'], lib="torch")
