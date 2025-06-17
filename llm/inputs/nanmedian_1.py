
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def nanmedian_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor with NaNs
    input1 = np.array([1.0, np.nan, 3.0, 2.0])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D tensor with NaNs
    input2 = np.array([[2.0, 3.0, 1.0], [np.nan, 1.0, np.nan]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 3D tensor with NaNs, including negative values
    input3 = np.array([[[1.0, np.nan, -2.0], [3.0, -1.0, np.nan]], [[np.nan, 2.0, 1.0], [-3.0, np.nan, 4.0]]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: All NaNs
    input4 = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: Integer tensor with NaNs (converted to float)
    input5 = np.array([1, np.nan, 3, 2]).astype(float)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Case 6: Larger tensor with mixed positive, negative, and NaN values
    input6 = np.array([[-1.0, 2.0, np.nan, 4.0], [5.0, np.nan, -6.0, 7.0], [np.nan, 8.0, 9.0, np.nan]])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: 1D tensor with only one non-NaN value
    input7 = np.array([np.nan, np.nan, 5.0, np.nan, np.nan])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nanmedian_1"] = nanmedian_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nanmedian_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nanmedian_1'.")

check_valid('torch.nanmedian', generated_inputs['torch.nanmedian_1'], lib="torch")
