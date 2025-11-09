
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def flipud_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor
    input = torch.arange(4).view(2, 2).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor
    input = torch.arange(5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor
    input = torch.arange(24).view(2, 3, 4).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor
    input = torch.arange(16).view(2, 2, 2, 2).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with negative values
    input = torch.tensor([[-1.0, -2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with float values
    input = torch.tensor([[1.5, 2.5], [3.5, 4.5]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large 2D tensor
    input = torch.arange(100).view(10, 10).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with single element
    input = torch.tensor([42]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with single row
    input = torch.tensor([[1, 2, 3]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with single column
    input = torch.tensor([[1], [2], [3]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 3D tensor with negative values
    input = torch.tensor([[[-1, -2], [-3, -4]], [[5, 6], [7, 8]]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: 2D tensor with mixed positive/negative/zero
    input = torch.tensor([[0, -1, 2], [-3, 4, -5]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.flipud"] = flipud_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.flipud' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.flipud'.")


check_valid('torch.flipud', generated_inputs['torch.flipud'], lib="torch", suffix=0)
