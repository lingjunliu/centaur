
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def permute_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(2, 3, 5).numpy()
    dims = (2, 0, 1)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(1, 4, 6, 8).numpy()
    dims = (3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(7, 2).numpy()
    dims = (1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(3, 3, 3, 3).numpy()
    dims = (3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(2, 5, 7, 3, 6).numpy()
    dims = (4, 3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(1, 1).numpy()
    dims = (1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(4, 5, 6).numpy()
    dims = (2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(1, 2, 3, 4, 5).numpy()
    dims = (4, 3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(6, 4, 2).numpy()
    dims = (2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(1, 2, 3, 4, 5, 6).numpy()
    dims = (5, 4, 3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.permute"] = permute_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.permute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.permute'.")


check_valid('torch.permute', generated_inputs['torch.permute'], lib="torch", suffix=0)
