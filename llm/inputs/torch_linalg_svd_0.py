
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_svd_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    A = np.random.rand(3, 3).astype(np.float32)
    input_dict = {"A": A, "full_matrices": True, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rectangular matrix (m > n)
    A = np.random.rand(5, 3).astype(np.float32)
    input_dict = {"A": A, "full_matrices": False, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix (m < n)
    A = np.random.rand(3, 5).astype(np.float32)
    input_dict = {"A": A, "full_matrices": True, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of matrices
    A = np.random.rand(2, 4, 4).astype(np.float32)
    input_dict = {"A": A, "full_matrices": False, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex matrix
    A = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(np.complex64)
    input_dict = {"A": A, "full_matrices": True, "driver": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.svd"] = linalg_svd_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.svd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.svd'.")

check_valid('torch.linalg.svd', generated_inputs['torch.linalg.svd'], lib="torch", suffix=0)
