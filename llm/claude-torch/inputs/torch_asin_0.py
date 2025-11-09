
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def asin_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([0.0, 0.5, -0.5, 0.707])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[0.1, 0.2], [0.3, 0.4]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array(0.5)
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[-0.1, -0.2], [-0.3, -0.4]], [[0.1, 0.2], [0.3, 0.4]]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1.0, -0.9, 0.9, 1.0])
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
    
    input_tensor = np.array([0.1, 0.2, 0.3])
    out_tensor = np.empty(3)
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[0.5, -0.5], [0.3, -0.3]])
    out_tensor = np.empty((2, 2))
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.linspace(-0.99, 0.99, 100)
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.01, 0.001, 0.0001])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[[0.1, 0.2], [0.3, 0.4]]]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.asin"] = asin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asin'.")


check_valid('torch.asin', generated_inputs['torch.asin'], lib="torch", suffix=0)
