
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def positive_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor with positive and negative values
    input = torch.randn(5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor
    input = torch.randn(3, 4).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor
    input = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: scalar tensor
    input = torch.tensor(5.0).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: tensor with all negative values
    input = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: tensor with all positive values
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: tensor with zeros
    input = torch.zeros(4, 5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: tensor with ones
    input = torch.ones(3, 3).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor
    input = torch.randn(2, 2, 2, 2).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: large 1D tensor
    input = torch.randn(100).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: integer tensor
    input = torch.tensor([1, -2, 3, -4, 5]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: float64 tensor
    input = torch.tensor([1.5, -2.5, 3.5], dtype=torch.float64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.positive"] = positive_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.positive' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.positive'.")


check_valid('torch.positive', generated_inputs['torch.positive'], lib="torch", suffix=0)
