
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def any_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([False, True])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[False, False], [False, False]])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0, 0, 0])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0, 1, 2])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.0, 0.0, 0.0])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.0, 1.5, -2.3])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[True, False], [False, True]], [[False, False], [True, True]]])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1, -2, -3])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([True])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.zeros((10, 10), dtype=bool)
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0, 1, 2], dtype=np.uint8)
    out = np.array(0, dtype=np.uint8)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.any_1"] = any_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.any_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.any_1'.")


check_valid('torch.any', generated_inputs['torch.any_1'], lib="torch", suffix=1)
