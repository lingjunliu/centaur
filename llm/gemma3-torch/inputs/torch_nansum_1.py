
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nansum_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    dtype1 = torch.float64
    input_dict1 = {"input": input1, "dtype": dtype1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[-1.0, np.nan, 3.0], [4.0, -np.inf, 6.0]], dtype=np.float32)
    dtype2 = torch.float16
    input_dict2 = {"input": input2, "dtype": dtype2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    dtype3 = torch.float32
    input_dict3 = {"input": input3, "dtype": dtype3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[np.nan, 2.0], [3.0, np.nan]], dtype=np.float16)
    dtype4 = torch.float32
    input_dict4 = {"input": input4, "dtype": dtype4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    dtype5 = None
    input_dict5 = {"input": input5, "dtype": dtype5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [np.nan, 8.0]]], dtype=np.float32)
    dtype6 = torch.float64
    input_dict6 = {"input": input6, "dtype": dtype6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([-1.0, -2.0, np.nan, -4.0], dtype=np.float32)
    dtype7 = torch.float16
    input_dict7 = {"input": input7, "dtype": dtype7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dtype8 = torch.float32
    input_dict8 = {"input": input8, "dtype": dtype8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    dtype9 = torch.float64
    input_dict9 = {"input": input9, "dtype": dtype9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    dtype10 = torch.float64
    input_dict10 = {"input": input10, "dtype": dtype10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nansum_1"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_1'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_1'], lib="torch", suffix=1)
