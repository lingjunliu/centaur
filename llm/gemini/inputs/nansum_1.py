
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def nansum_inputs():
    list_of_inputs = []

    # Case 1: 1D float tensor with NaN
    input_tensor = np.array([1.0, 2.0, np.nan, 4.0])
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D int tensor with NaN
    input_tensor = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D float tensor with NaN and negative values
    input_tensor = np.array([[[1.0, np.nan], [-2.0, 3.0]], [[np.nan, 4.0], [-5.0, 6.0]]])
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D int tensor with NaN, specifying dtype
    input_tensor = np.array([1, 2, np.nan, 4], dtype=np.float32)
    input_dict = {"input": input_tensor, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: 2D tensor with only NaNs
    input_tensor = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 1D tensor without NaNs
    input_tensor = np.array([1, 2, 3, 4])
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Empty tensor
    input_tensor = np.array([])
    input_dict = {"input": input_tensor, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_1"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nansum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_1'.")

check_valid('torch.nansum', generated_inputs['torch.nansum_1'], lib="torch")
