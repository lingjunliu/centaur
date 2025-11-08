
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def replicationpad3d_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = 3
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(1, 1, 2, 3, 4).numpy()
    padding = 1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(2, 3, 5, 6, 7).numpy()
    padding = 2
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(4, 2, 10, 15, 20).numpy()
    padding = 4
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(8, 5, 1, 2, 3).numpy()
    padding = 5
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(1, 1, 10, 10, 10).numpy()
    padding = 0
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(3, 2, 5, 6, 7).numpy()
    padding = -1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(5, 3, 2, 3, 4).numpy()
    padding = 6
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(7, 1, 3, 2, 1).numpy()
    padding = 7
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(6, 4, 8, 9, 10).numpy()
    padding = 8
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_1"] = replicationpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad3d_1'.")


check_valid('torch.nn.ReplicationPad3d', generated_inputs['torch.nn.ReplicationPad3d_1'], lib="torch", suffix=1)
