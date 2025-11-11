
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def replication_pad_3d_inputs():
    list_of_inputs = []
    
    # Input 1: 5D tensor with padding tuple (3, 3, 6, 6, 1, 1)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (3, 3, 6, 6, 1, 1)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 5D tensor with padding tuple (1, 1, 1, 1, 1, 1)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (1, 1, 1, 1, 1, 1)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 5D tensor with padding tuple (0, 0, 0, 0, 0, 0)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (0, 0, 0, 0, 0, 0)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 5D tensor with padding tuple (2, 2, 3, 3, 4, 4)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (2, 2, 3, 3, 4, 4)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 5D tensor with padding tuple (1, 1, 2, 2, 3, 3)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (1, 1, 2, 2, 3, 3)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 5D tensor with padding tuple (5, 5, 10, 10, 15, 15)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (5, 5, 10, 10, 15, 15)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 5D tensor with padding tuple (1, 2, 3, 4, 5, 6)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (1, 2, 3, 4, 5, 6)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 5D tensor with padding tuple (0, 1, 2, 3, 4, 5)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (0, 1, 2, 3, 4, 5)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 5D tensor with padding tuple (3, 3, 0, 0, 0, 0)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (3, 3, 0, 0, 0, 0)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 5D tensor with padding tuple (1, 2, 3, 4, 5, 6)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (1, 2, 3, 4, 5, 6)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_2"] = replication_pad_3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad3d_2'.")


check_valid('torch.nn.ReplicationPad3d', generated_inputs['torch.nn.ReplicationPad3d_2'], lib="torch", suffix=2)
