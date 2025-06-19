
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def gradient_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0]).numpy()
    spacing = [np.array([1.0, 2.0, 3.0, 4.0])]
    dim = [0]
    edge_order = 1

    input_dict = {
        "input": input_tensor,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]]).numpy()
    spacing = [np.array([1.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0])]
    dim = [0, 1]
    edge_order = 2

    input_dict = {
        "input": input_tensor,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    spacing = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    dim = [0, 1, 2]
    edge_order = 1
    input_dict = {
        "input": input_tensor,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(5, 5).numpy()
    spacing = [np.arange(1, 6).astype(np.float64), np.arange(6, 11).astype(np.float64)]
    dim = [0, 1]
    edge_order = 2

    input_dict = {
        "input": input_tensor,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3).numpy()
    spacing = [np.array([2.0, 4.0, 6.0]), np.array([0.5, 1.0, 1.5])]
    dim = [0, 1]
    edge_order = 1

    input_dict = {
        "input": input_tensor,
        "spacing": spacing,
        "dim": dim,
        "edge_order": edge_order
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs
generated_inputs = {}
generated_inputs["torch.gradient"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gradient'.")

check_valid('torch.gradient', generated_inputs['torch.gradient'], lib="torch", suffix=0)
