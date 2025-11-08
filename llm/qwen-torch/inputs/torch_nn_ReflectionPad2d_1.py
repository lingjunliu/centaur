
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    # Input 1 - padding of 2, 3D tensor
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = 2
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - different padding values for each side, 3D tensor
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (1, 1, 2, 0)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - padding of 1, 4D tensor
    input = torch.arange(16, dtype=torch.float).reshape(1, 1, 4, 4).numpy()
    padding = 1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - negative padding value, 3D tensor
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = -1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - padding of 0, 2D tensor (no padding)
    input = torch.arange(4, dtype=torch.float).reshape(1, 1, 2, 2).numpy()
    padding = 0
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - padding of 3, 4D tensor
    input = torch.arange(16, dtype=torch.float).reshape(1, 1, 4, 4).numpy()
    padding = 3
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - different padding values for each side, 2D tensor
    input = torch.arange(4, dtype=torch.float).reshape(1, 1, 2, 2).numpy()
    padding = (0, 2, 1, 1)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - padding of 4, 3D tensor
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = 4
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - padding of 5, 4D tensor
    input = torch.arange(25, dtype=torch.float).reshape(1, 1, 5, 5).numpy()
    padding = 5
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - padding of 6, 3D tensor
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = 6
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_1"] = reflectionpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad2d_1'.")


check_valid('torch.nn.ReflectionPad2d', generated_inputs['torch.nn.ReflectionPad2d_1'], lib="torch", suffix=1)
