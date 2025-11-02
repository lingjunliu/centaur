
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with padding tuple
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (1, 1, 1, 1)
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Different padding values for each side
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (2, 1, 0, 3)
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - Single value padding (same padding on all sides)
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (2, 2, 2, 2)
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - Single value padding with zero padding
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (0, 0, 0, 0)
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - Multi-dimensional tensor with padding
    input = torch.arange(16, dtype=torch.float).reshape(1, 1, 4, 4).numpy()
    padding = (2, 1, 0, 3)
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Tensor with different shape
    input = torch.arange(16, dtype=torch.float).reshape(1, 1, 4, 4).numpy()
    padding = (3, 2, 1, 0)
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Large padding values with negative input values
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (3, 2, 1, 0)
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Padding values less than input dimensions
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (0, 1, 2, 3)
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Padding values with negative numbers
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (-1, -2, -3, -4)
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Padding values with zero values
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (0, 0, 0, 0)
    input_dict = {
        "input": input,
        "padding": padding
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
