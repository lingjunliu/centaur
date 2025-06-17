
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def isreal_inputs():
    list_of_inputs = []

    input_dict = {"input": np.array([1, 2, 3])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([1.0, 2.5, -3.2])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([1 + 0j, 2 + 0j, 3 + 1j])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([[1, 2], [3, 4]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([1 + 0j, 2 + 0j, 3 + 0j])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([1.0 + 0j, 2.5 + 0j, -3.2 + 0j])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([1+1j, 2+2j, 3+0j])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([0])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([1, 2, np.inf])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.isreal"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.isreal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isreal'.")

check_valid('torch.isreal', generated_inputs['torch.isreal'], lib="torch")
