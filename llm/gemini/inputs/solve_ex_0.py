
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def solve_ex_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors, left=True
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0], [6.0]])
    input_dict = {"A": A, "B": B, "left": True, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Basic float tensors, left=False
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0]])
    input_dict = {"A": A, "B": B, "left": False, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Complex tensors, left=True
    A = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]])
    B = np.array([[5.0 + 2j], [6.0]])
    input_dict = {"A": A, "B": B, "left": True, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different shaped tensors (A is 3x3, B is 3x1), left=True. Make A non-singular
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    B = np.array([[10.0], [11.0], [12.0]])
    input_dict = {"A": A, "B": B, "left": True, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5:  float tensors, check_errors=False
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0], [6.0]])
    input_dict = {"A": A, "B": B, "left": True, "check_errors": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.solve_ex"] = solve_ex_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.solve_ex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.solve_ex'.")

check_valid('torch.linalg.solve_ex', generated_inputs['torch.linalg.solve_ex'], lib="torch")
