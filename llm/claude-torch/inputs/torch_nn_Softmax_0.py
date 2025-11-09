
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def softmax_inputs():
    list_of_inputs = []
    
    input_tensor = np.random.randn(2, 3).astype(np.float32)
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "dim": 0,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    input_dict = {
        "dim": 0,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "dim": 2,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 5, 7).astype(np.float32)
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "dim": 3,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]).astype(np.float32)
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 6).astype(np.float32)
    input_dict = {
        "dim": -1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {
        "dim": -2,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(1, 2, 3, 2, 4).astype(np.float32)
    input_dict = {
        "dim": 4,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(10, 100).astype(np.float32)
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softmax"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softmax'.")


check_valid('torch.nn.Softmax', generated_inputs['torch.nn.Softmax'], lib="torch", suffix=0)
