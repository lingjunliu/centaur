
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def is_nonzero_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor(1.0).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor(0.0).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor(-1.0).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((1, 1)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.zeros((1, 1)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_nonzero"] = is_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_nonzero'.")


check_valid('torch.is_nonzero', generated_inputs['torch.is_nonzero'], lib="torch", suffix=0)
