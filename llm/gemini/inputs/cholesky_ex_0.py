
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cholesky_ex_inputs():
    list_of_inputs = []

    # Example 1: Basic positive definite matrix
    A = np.array([[4, 12], [12, 37]], dtype=np.float64)
    input_dict = {"A": A, "upper": False, "check_errors": True}
    list_of_inputs.append(input_dict)

    # Example 2: Upper triangular
    A = np.array([[16, 0], [0, 9]], dtype=np.float64)
    input_dict = {"A": A, "upper": True, "check_errors": True}
    list_of_inputs.append(input_dict)

    # Example 3: Batched input
    A = np.array([[[4, 0], [0, 1]], [[1, 0], [0, 4]]], dtype=np.float64)
    input_dict = {"A": A, "upper": False, "check_errors": True}
    list_of_inputs.append(input_dict)

    # Example 4: Single element matrix
    A = np.array([[25]], dtype=np.float64)
    input_dict = {"A": A, "upper": False, "check_errors": True}
    list_of_inputs.append(input_dict)

    # Example 5: Small matrix with different dtype
    A = np.array([[1, 0], [0, 1]], dtype=np.float32)
    input_dict = {"A": A, "upper": False, "check_errors": True}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.cholesky_ex"] = cholesky_ex_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.cholesky_ex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.cholesky_ex'.")

check_valid('torch.linalg.cholesky_ex', generated_inputs['torch.linalg.cholesky_ex'], lib="torch")
