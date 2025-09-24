
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cholesky_inverse_inputs():
    list_of_inputs = []

    # Example 1: Basic positive definite matrix
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float32)
    L = np.linalg.cholesky(A)
    input_dict = {"input": L}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Different size matrix
    A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)
    L = np.linalg.cholesky(A)
    input_dict = {"input": L}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Batch of matrices
    A1 = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float32)
    L1 = np.linalg.cholesky(A1)
    A2 = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float32)
    L2 = np.linalg.cholesky(A2)
    batch = np.stack([L1, L2])
    input_dict = {"input": batch}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Single element matrix
    A = np.array([[4]], dtype=np.float32)
    L = np.linalg.cholesky(A)
    input_dict = {"input": L}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Larger matrix
    A = np.array([[10, 2, 3, 1], [2, 11, 4, 2], [3, 4, 12, 3], [1, 2, 3, 13]], dtype=np.float32)
    L = np.linalg.cholesky(A)
    input_dict = {"input": L}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Using Float64
    A = np.array([[4, 1], [1, 4]], dtype=np.float64)
    L = np.linalg.cholesky(A)
    input_dict = {"input": L}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 7: Batch of matrices with Float64
    A1 = np.array([[4, 1], [1, 4]], dtype=np.float64)
    L1 = np.linalg.cholesky(A1)
    A2 = np.array([[9, 2], [2, 9]], dtype=np.float64)
    L2 = np.linalg.cholesky(A2)
    batch = np.stack([L1, L2])
    input_dict = {"input": batch}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cholesky_inverse"] = cholesky_inverse_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cholesky_inverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cholesky_inverse'.")

check_valid('torch.cholesky_inverse', generated_inputs['torch.cholesky_inverse'], lib="torch")
