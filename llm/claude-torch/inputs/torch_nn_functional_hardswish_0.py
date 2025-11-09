
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def hardswish_inputs():
    list_of_inputs = []
    
    input_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([-5.0, -3.0, -1.0, 0.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[-4.0, -2.0], [1.0, 3.5]], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.0, -1.0], [-2.0, -3.0]]], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([-3.5, -3.0, -2.5, 2.5, 3.0, 3.5], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.zeros((3, 3), dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([2.5], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([-10.0, -20.0, -100.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([10.0, 20.0, 100.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardswish"] = hardswish_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.hardswish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardswish'.")


check_valid('torch.nn.functional.hardswish', generated_inputs['torch.nn.functional.hardswish'], lib="torch", suffix=0)
