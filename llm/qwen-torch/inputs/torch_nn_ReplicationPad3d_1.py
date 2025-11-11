
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def replication_pad3d_inputs():
    list_of_inputs = []
    
    # Input 1: padding = 3, input is 5D tensor
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = 3
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: padding = (3, 3, 6, 6, 1, 1), input is 5D tensor
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (3, 3, 6, 6, 1, 1)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: padding = 1, input is 4D tensor
    input = torch.randn(2, 3, 320, 480).numpy()
    padding = 1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: padding = (1, 2, 3, 4, 5, 6), input is 4D tensor
    input = torch.randn(2, 3, 320, 480).numpy()
    padding = (1, 2, 3, 4, 5, 6)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: padding = 0, input is 3D tensor
    input = torch.randn(2, 3, 320).numpy()
    padding = 0
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: padding = (0, 0, 0, 0, 0, 0), input is 3D tensor
    input = torch.randn(2, 3, 320).numpy()
    padding = (0, 0, 0, 0, 0, 0)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: padding = -1, input is 5D tensor
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = -1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: padding = (-1, -1, -1, -1, -1, -1), input is 5D tensor
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (-1, -1, -1, -1, -1, -1)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: padding = 5, input is 4D tensor with different shape
    input = torch.randn(2, 3, 100, 200).numpy()
    padding = 5
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: padding = (5, 4, 3, 2, 1, 0), input is 4D tensor with different shape
    input = torch.randn(2, 3, 100, 200).numpy()
    padding = (5, 4, 3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_1"] = replication_pad3d_inputs()

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
