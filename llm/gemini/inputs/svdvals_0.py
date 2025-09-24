
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def linalg_svdvals_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensor
    A = np.random.randn(5, 3).astype(np.float32)
    input_dict = {"A": A, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Double tensor
    A = np.random.randn(4, 4).astype(np.float64)
    input_dict = {"A": A, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Complex tensor
    A = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(np.complex64)
    input_dict = {"A": A, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Complex double tensor
    A = (np.random.randn(2, 5) + 1j * np.random.randn(2, 5)).astype(np.complex128)
    input_dict = {"A": A, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Batched tensor
    A = np.random.randn(2, 5, 3).astype(np.float32)
    input_dict = {"A": A, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.svdvals"] = linalg_svdvals_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.svdvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.svdvals'.")

check_valid('torch.linalg.svdvals', generated_inputs['torch.linalg.svdvals'], lib="torch")
