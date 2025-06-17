
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def matrix_rank_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensor
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict1 = {"input": input1, "atol": 1e-05, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Singular matrix
    input2 = np.array([[1.0, 2.0], [2.0, 4.0]])
    input_dict2 = {"input": input2, "atol": 1e-05, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 3D tensor
    input3 = np.random.rand(2, 3, 4)
    input_dict3 = {"input": input3, "atol": 1e-05, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Complex tensor
    input4 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    input_dict4 = {"input": input4, "atol": 1e-05, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Tall matrix
    input5 = np.random.rand(5, 2)
    input_dict5 = {"input": input5, "atol": 1e-05, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

def convert_to_torch(item):
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item).float()
    return item
    

def modify_input_dict(input_dict):
    new_input_dict = {}
    for k, v in input_dict.items():
        new_input_dict[k] = convert_to_torch(v)
    return new_input_dict

modified_list_of_inputs = []
for input_dict in matrix_rank_inputs():
    modified_list_of_inputs.append(modify_input_dict(input_dict))
    
generated_inputs["torch.linalg.matrix_rank_1"] = modified_list_of_inputs

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_rank_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_rank_1'.")

check_valid('torch.linalg.matrix_rank', generated_inputs['torch.linalg.matrix_rank_1'], lib="torch")
