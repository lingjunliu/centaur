
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def floor_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([-0.8166, 1.5308, -0.2530, -0.2091])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.5, -2.3, 3.7], [-4.1, 5.9, -6.2]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array(3.14159)
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-100.9, -200.1, -300.5])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.001, -0.001, 0.999, -0.999])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.5, 2.5, 3.5])
    out_tensor = np.empty(3)
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[[1.1, 2.2], [3.3, 4.4]]]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.0, 0.0, 0.0])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([999.99, 1000.01, 5000.5])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor'.")


check_valid('torch.floor', generated_inputs['torch.floor'], lib="torch", suffix=0)
