
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cholesky_solve_inputs():
    list_of_inputs = []

    # Input 1: Basic test case with float32
    A = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]], dtype=np.float32)
    B = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    L = np.linalg.cholesky(A).astype(np.float32)
    input_dict = {"input": B, "L": L, "upper": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64
    A = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]], dtype=np.float64)
    B = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    L = np.linalg.cholesky(A).astype(np.float64)
    input_dict = {"input": B, "L": L, "upper": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Multiple right-hand sides
    A = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]], dtype=np.float32)
    B = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
    L = np.linalg.cholesky(A).astype(np.float32)
    input_dict = {"input": B, "L": L, "upper": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched inputs
    A = np.array([[[4.0, 12.0], [12.0, 37.0]], [[98.0, -43.0], [-43.0, 98.0]]], dtype=np.float32)
    B = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    L = np.linalg.cholesky(A).astype(np.float32)
    input_dict = {"input": B, "L": L, "upper": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Test with upper=True
    A = np.array([[4.0, 12.0, -16.0], [0.0, 37.0, -43.0], [0.0, 0.0, 98.0]], dtype=np.float32)
    B = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    U = np.linalg.cholesky(A.T).T.astype(np.float32)
    input_dict = {"input": B, "L": U, "upper": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cholesky_solve"] = cholesky_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cholesky_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cholesky_solve'.")

check_valid('torch.cholesky_solve', generated_inputs['torch.cholesky_solve'], lib="torch")
