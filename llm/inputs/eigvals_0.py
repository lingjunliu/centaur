
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def linalg_eigvals_inputs():
    list_of_inputs = []

    A = np.random.rand(2, 2).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(3, 3).astype(np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = (np.random.rand(2, 2) + 1j * np.random.rand(2, 2)).astype(np.complex64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(np.complex128)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1, 2], [3, 4]]).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.eigvals"] = linalg_eigvals_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.eigvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eigvals'.")

check_valid('torch.linalg.eigvals', generated_inputs['torch.linalg.eigvals'], lib="torch")
