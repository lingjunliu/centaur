
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def logsigmoid_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor with positive values
    input_dict = {
        "input": torch.randn(5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor with negative values
    input_dict = {
        "input": torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor
    input_dict = {
        "input": torch.randn(3, 4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    input_dict = {
        "input": torch.randn(2, 3, 4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor
    input_dict = {
        "input": torch.randn(2, 3, 4, 5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Single scalar value
    input_dict = {
        "input": torch.tensor([5.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Tensor with zeros
    input_dict = {
        "input": torch.zeros(3, 3).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Tensor with mixed positive and negative values
    input_dict = {
        "input": torch.tensor([[-1.5, 2.3], [0.5, -3.2]]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large values
    input_dict = {
        "input": torch.tensor([10.0, 20.0, 30.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Very small values
    input_dict = {
        "input": torch.tensor([0.01, 0.001, 0.0001]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 5D tensor
    input_dict = {
        "input": torch.randn(1, 2, 2, 2, 2).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Single element tensor
    input_dict = {
        "input": torch.tensor([-0.5]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSigmoid"] = logsigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LogSigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LogSigmoid'.")


check_valid('torch.nn.LogSigmoid', generated_inputs['torch.nn.LogSigmoid'], lib="torch", suffix=0)
