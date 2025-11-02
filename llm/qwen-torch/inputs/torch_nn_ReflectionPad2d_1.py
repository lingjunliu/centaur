
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]]], dtype=torch.float).numpy()
    padding = 2
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]]], dtype=torch.float).numpy()
    padding = (1, 1, 2, 0)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[[[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]]]], dtype=torch.float).numpy()
    padding = 1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.], [10., 11., 12., 13., 14.]]]], dtype=torch.float).numpy()
    padding = (2, 2, 1, 1)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[[0., 1., 2., 3., 4., 5.], [6., 7., 8., 9., 10., 11.], [12., 13., 14., 15., 16., 17.]]]], dtype=torch.float).numpy()
    padding = 3
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[[[0., 1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12., 13.], [14., 15., 16., 17., 18., 19., 20.]]]], dtype=torch.float).numpy()
    padding = (0, 0, 2, 2)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[[[0., 1., 2., 3., 4., 5., 6., 7.], [8., 9., 10., 11., 12., 13., 14., 15.]]]], dtype=torch.float).numpy()
    padding = (1, 2, 3, 4)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[[0., 1., 2., 3., 4., 5., 6., 7., 8.], [9., 10., 11., 12., 13., 14., 15., 16., 17.]]]], dtype=torch.float).numpy()
    padding = (2, 2, 2, 2)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], [10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]]]], dtype=torch.float).numpy()
    padding = (0, 1, 2, 3)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[[[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], [11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.]]]], dtype=torch.float).numpy()
    padding = (3, 3, 3, 3)
    
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
