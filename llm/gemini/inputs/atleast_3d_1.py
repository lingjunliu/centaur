
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def atleast_3d_inputs():
    list_of_inputs = []

    # Case 1: Scalar
    input_1 = np.array(5).astype(np.float32)
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 1D array
    input_2 = np.array([1, 2, 3]).astype(np.int64)
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 2D array
    input_3 = np.array([[1, 2], [3, 4]]).astype(np.float64)
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 3D array
    input_4 = np.random.rand(2, 3, 4).astype(np.complex64)
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: 4D array
    input_5 = np.random.randn(1, 2, 3, 4).astype(np.float32)
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Negative values
    input_6 = np.array([-1, -2, -3]).astype(np.int32)
    input_dict_6 = {"input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Empty array
    input_7 = np.array([]).astype(np.float32)
    input_dict_7 = {"input": input_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["torch.atleast_3d_1"] = atleast_3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.atleast_3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atleast_3d_1'.")

check_valid('torch.atleast_3d', generated_inputs['torch.atleast_3d_1'], lib="torch")
