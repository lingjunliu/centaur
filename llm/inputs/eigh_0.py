
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def eigh_inputs():
    list_of_inputs = []

    A = np.array([[2., 1.], [1., 2.]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": "L", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[2., 1.], [1., 2.]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": "U", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1., 2.], [2., 1.]], dtype=np.float32)
    input_dict = {"A": A, "UPLO": "L", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1., 2.], [2., 1.]], dtype=np.float32)
    input_dict = {"A": A, "UPLO": "U", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = np.array([[2 + 1j, 1 - 1j], [1 + 1j, 2 - 1j]], dtype=np.complex128)
    input_dict = {"A": A, "UPLO": "L", "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.eigh"] = eigh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.eigh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eigh'.")

check_valid('torch.linalg.eigh', generated_inputs['torch.linalg.eigh'], lib="torch")
