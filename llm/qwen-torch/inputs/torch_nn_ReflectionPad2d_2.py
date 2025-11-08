
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    # Input 1: padding as int, input tensor 3D
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (2, 2, 2, 2)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: padding as tuple, input tensor 4D
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (1, 1, 2, 0)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: padding as int, input tensor 2D
    input = torch.arange(4, dtype=torch.float).reshape(1, 1, 2, 2).numpy()
    padding = (1, 1, 1, 1)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: padding as tuple, input tensor 3D with negative values
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (0, 0, -1, -1)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: padding as int, input tensor 4D with different sizes
    input = torch.arange(16, dtype=torch.float).reshape(1, 1, 4, 4).numpy()
    padding = (3, 3, 3, 3)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: padding as tuple, input tensor 3D with different padding values
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (2, 1, 0, 3)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: padding as int, input tensor 1D
    input = torch.arange(3, dtype=torch.float).reshape(1, 1, 3).numpy()
    padding = (2,)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: padding as tuple, input tensor 4D with large padding values
    input = torch.arange(16, dtype=torch.float).reshape(1, 1, 4, 4).numpy()
    padding = (2, 3, 4, 5)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: padding as int, input tensor 3D with zero padding
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (0, 0, 0, 0)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: padding as tuple, input tensor 4D with float values in input
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (2, 2, 1, 1)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_2"] = reflectionpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad2d_2'.")


check_valid('torch.nn.ReflectionPad2d', generated_inputs['torch.nn.ReflectionPad2d_2'], lib="torch", suffix=2)
