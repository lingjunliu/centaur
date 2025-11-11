
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with padding as int
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (2, 2, 2, 2)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Different paddings for different sides
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    padding = (1, 1, 2, 0)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: With negative values in input tensor
    input = torch.tensor([[-1., -2., -3.], [4., 5., 6.], [7., 8., 9.]]).numpy()
    padding = (1, 1, 1, 1)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: With different dimensions
    input = torch.ones((2, 3, 4, 5)).numpy()
    padding = (0, 1, 2, 3)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: With zero padding (no change in output size)
    input = torch.arange(4, dtype=torch.float).reshape(1, 1, 2, 2).numpy()
    padding = (0, 0, 0, 0)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: With large padding values
    input = torch.ones((1, 1, 3, 3)).numpy()
    padding = (5, 5, 5, 5)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: With single dimension input tensor
    input = torch.arange(3, dtype=torch.float).reshape(1, 1, 1, 3).numpy()
    padding = (1, 1, 1, 1)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: With floating point values in input tensor
    input = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy()
    padding = (0, 1, 1, 0)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: With large input tensor dimensions
    input = torch.ones((1, 1, 10, 10)).numpy()
    padding = (3, 3, 3, 3)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: With mixed positive and negative values
    input = torch.tensor([[1., -2.], [-3., 4.]]).numpy()
    padding = (1, 1, 1, 1)
    
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
