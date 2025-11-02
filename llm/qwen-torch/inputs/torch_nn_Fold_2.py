
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def fold_inputs():
    list_of_inputs = []
    
    # Input 1 - Valid
    input = torch.randn(1, 3 * 2 * 2, 12).numpy()
    input_dict = {
        "output_size": (4, 5),
        "kernel_size": (2, 2),
        "dilation": (1, 1),
        "padding": (0, 0),
        "stride": (1, 1),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Valid
    input = torch.randn(2, 3 * 3 * 3, 18).numpy()
    input_dict = {
        "output_size": (5, 6),
        "kernel_size": (3, 3),
        "dilation": (1, 1),
        "padding": (1, 1),
        "stride": (1, 1),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - Valid
    input = torch.randn(1, 3 * 2 * 2, 12).numpy()
    input_dict = {
        "output_size": (4, 5),
        "kernel_size": (2, 2),
        "dilation": (2, 2),
        "padding": (0, 0),
        "stride": (1, 1),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - Valid
    input = torch.randn(1, 3 * 4 * 4, 24).numpy()
    input_dict = {
        "output_size": (6, 7),
        "kernel_size": (4, 4),
        "dilation": (1, 1),
        "padding": (0, 0),
        "stride": (2, 2),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - Valid
    input = torch.randn(3, 3 * 5 * 5, 30).numpy()
    input_dict = {
        "output_size": (7, 8),
        "kernel_size": (5, 5),
        "dilation": (2, 2),
        "padding": (1, 1),
        "stride": (2, 2),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Valid
    input = torch.randn(1, 3 * 2 * 2, 12).numpy()
    input_dict = {
        "output_size": (4, 5),
        "kernel_size": (2, 2),
        "dilation": (1, 1),
        "padding": (2, 2),
        "stride": (1, 1),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Valid
    input = torch.randn(4, 3 * 3 * 3, 18).numpy()
    input_dict = {
        "output_size": (5, 6),
        "kernel_size": (3, 3),
        "dilation": (2, 2),
        "padding": (0, 0),
        "stride": (2, 2),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Valid
    input = torch.randn(1, 3 * 6 * 6, 36).numpy()
    input_dict = {
        "output_size": (8, 9),
        "kernel_size": (6, 6),
        "dilation": (1, 1),
        "padding": (0, 0),
        "stride": (1, 1),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Valid
    input = torch.randn(2, 3 * 7 * 7, 49).numpy()
    input_dict = {
        "output_size": (9, 10),
        "kernel_size": (7, 7),
        "dilation": (2, 2),
        "padding": (1, 1),
        "stride": (3, 3),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Valid
    input = torch.randn(1, 3 * 8 * 8, 64).numpy()
    input_dict = {
        "output_size": (10, 11),
        "kernel_size": (8, 8),
        "dilation": (1, 1),
        "padding": (0, 0),
        "stride": (1, 1),
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Fold_2"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Fold_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Fold_2'.")


check_valid('torch.nn.Fold', generated_inputs['torch.nn.Fold_2'], lib="torch", suffix=2)
