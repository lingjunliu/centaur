
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def gradient_inputs():
    list_of_inputs = []
    
    input_arr = np.array([4., 1., 1., 16.])
    spacing = (np.array([-2., -1., 1., 4.]),)
    dim = [0]
    edge_order = 1
    input_dict = {
        "input": input_arr,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[1., 2., 4., 8.], [10., 20., 40., 80.]])
    spacing = (np.array([0., 2.]), np.array([0., 3., 6., 9.]))
    dim = [0, 1]
    edge_order = 1
    input_dict = {
        "input": input_arr,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[1., 2., 4., 8.], [10., 20., 40., 80.]])
    spacing = (np.array([0., 1., 2., 3.]),)
    dim = [1]
    edge_order = 1
    input_dict = {
        "input": input_arr,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.random.randn(3, 4, 5)
    spacing = (np.array([0., 1., 2.]), np.array([0., 1., 2., 3.]), np.array([0., 1., 2., 3., 4.]))
    dim = [0, 1, 2]
    edge_order = 2
    input_dict = {
        "input": input_arr,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([-5., -2., 1., 4., 10.])
    spacing = (np.array([0., 1., 2., 3., 4.]),)
    dim = [0]
    edge_order = 1
    input_dict = {
        "input": input_arr,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[1., 4., 9., 16.], [2., 8., 18., 32.]])
    spacing = (np.array([0., 5.]), np.array([1., 2., 3., 4.]))
    dim = [0, 1]
    edge_order = 2
    input_dict = {
        "input": input_arr,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.random.randn(2, 3, 4)
    spacing = (np.array([0., 1., 2.]), np.array([0., 1., 2., 3.]))
    dim = [1, 2]
    edge_order = 1
    input_dict = {
        "input": input_arr,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.gradient_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gradient_6'.")


check_valid('torch.gradient', generated_inputs['torch.gradient_6'], lib="torch", suffix=6)
