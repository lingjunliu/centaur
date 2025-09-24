
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def is_nonzero_inputs():
    list_of_inputs = []

    # Scalar Tensor
    input_dict = {"input": np.array(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Scalar Tensor - zero
    input_dict = {"input": np.array(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 1D Tensor
    input_dict = {"input": np.array([1])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 1D Tensor - nonzero
    input_dict = {"input": np.array([-1])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Float Tensor
    input_dict = {"input": np.array(1.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Negative Float Tensor
    input_dict = {"input": np.array(-1.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Zero Float Tensor
    input_dict = {"input": np.array(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_nonzero"] = is_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.is_nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_nonzero'.")

check_valid('torch.is_nonzero', generated_inputs['torch.is_nonzero'], lib="torch")
