
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def gradient_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([4., 1., 1., 16.]),
        "spacing": [1.0],
        "dim": [0],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1., 2., 4., 8.], [10., 20., 40., 80.]]),
        "spacing": [3., 2.],
        "dim": [0, 1],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1., 2., 4., 8.], [10., 20., 40., 80.], [5., 10., 20., 40.]]),
        "spacing": [2.0, 3.0],
        "dim": [0, 1],
        "edge_order": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1., 4., 9., 16., 25.]),
        "spacing": [1.],
        "dim": [0],
        "edge_order": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(3, 4, 5),
        "spacing": [1., 2., 0.5],
        "dim": [0, 1, 2],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1., 2., 4., 8.], [10., 20., 40., 80.]]),
        "spacing": [1.],
        "dim": [1],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.linspace(0, 10, 20),
        "spacing": [0.5],
        "dim": [0],
        "edge_order": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[5., 10., 15., 20.], [20., 25., 30., 35.], [40., 45., 50., 55.]]),
        "spacing": [-1., 2.],
        "dim": [0, 1],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(3, 4, 5, 6),
        "spacing": [1., 1., 1., 1.],
        "dim": [0, 1, 2, 3],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1., 3., 7., 12.], [2., 5., 11., 18.], [4., 8., 15., 24.]]),
        "spacing": [1.5, 2.5],
        "dim": [0, 1],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.gradient_5"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.gradient_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gradient_5'.")


check_valid('torch.gradient', generated_inputs['torch.gradient_5'], lib="torch", suffix=5)
