
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def atleast_3d_inputs():
    list_of_inputs = []

    # Scalar input
    input_scalar = np.array(5.0)
    input_dict = {"input": [input_scalar]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 1D input
    input_1d = np.array([1, 2, 3])
    input_dict = {"input": [input_1d]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2D input
    input_2d = np.array([[1, 2], [3, 4]])
    input_dict = {"input": [input_2d]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3D input
    input_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"input": [input_3d]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Higher dimension input (4D)
    input_4d = np.random.rand(2, 3, 4, 5)
    input_dict = {"input": [input_4d]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Negative values
    input_neg = np.array([-1, -2, -3])
    input_dict = {"input": [input_neg]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Float type
    input_float = np.array([1.5, 2.5, 3.5])
    input_dict = {"input": [input_float]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Int type
    input_int = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"input": [input_int]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # List of tensors
    input_list = [np.array([1, 2]), np.array([3, 4])]
    input_dict = {"input": input_list}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.atleast_3d_2"] = atleast_3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.atleast_3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atleast_3d_2'.")

check_valid('torch.atleast_3d', generated_inputs['torch.atleast_3d_2'], lib="torch")
