
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def matrix_rank_inputs():
    list_of_inputs = []

    def create_input_dict(input_array, tol=None, rtol=None, hermitian=False):
        input_tensor = torch.from_numpy(input_array)
        input_dict = {
            "input": input_tensor,
            "hermitian": hermitian
        }
        if tol is not None:
            input_dict["atol"] = tol
        if rtol is not None:
            input_dict["rtol"] = rtol
        return input_dict

    input1 = np.array([[1, 2], [2, 4]])
    list_of_inputs.append(create_input_dict(input1, tol=1e-8))

    input2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    list_of_inputs.append(create_input_dict(input2, tol=1e-8))

    input3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    list_of_inputs.append(create_input_dict(input3, tol=1e-8))

    input4 = np.array([[1, 2], [3, 4], [5, 6]])
    list_of_inputs.append(create_input_dict(input4, tol=1e-8))

    input5 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
    list_of_inputs.append(create_input_dict(input5, tol=1e-8))

    return list_of_inputs

generated_inputs["torch.linalg.matrix_rank_1"] = matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_rank_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_rank_1'.")

check_valid('torch.linalg.matrix_rank', generated_inputs['torch.linalg.matrix_rank_1'], lib="torch")
