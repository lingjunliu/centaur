
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def fold_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(1, 3 * 2 * 2, 12).numpy()
    input_dict = {
        "output_size": (4, 5),
        "kernel_size": (2, 2),
        "dilation": 1,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(2, 3 * 3 * 3, 16).numpy()
    input_dict = {
        "output_size": (5, 6),
        "kernel_size": (3, 3),
        "dilation": 1,
        "padding": 1,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(1, 3 * 2 * 2, 12).numpy()
    input_dict = {
        "output_size": (4, 5),
        "kernel_size": (2, 2),
        "dilation": 1,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(1, 3 * 4 * 4, 25).numpy()
    input_dict = {
        "output_size": (6, 7),
        "kernel_size": (4, 4),
        "dilation": 2,
        "padding": 1,
        "stride": 2,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(3, 3 * 2 * 2, 12).numpy()
    input_dict = {
        "output_size": (4, 5),
        "kernel_size": (2, 2),
        "dilation": 1,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(1, 3 * 5 * 5, 25).numpy()
    input_dict = {
        "output_size": (7, 8),
        "kernel_size": (5, 5),
        "dilation": 1,
        "padding": 2,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(1, 3 * 3 * 3, 16).numpy()
    input_dict = {
        "output_size": (5, 6),
        "kernel_size": (3, 3),
        "dilation": 2,
        "padding": 1,
        "stride": 2,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(1, 3 * 2 * 2, 12).numpy()
    input_dict = {
        "output_size": (4, 5),
        "kernel_size": (2, 2),
        "dilation": 1,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(1, 3 * 6 * 6, 36).numpy()
    input_dict = {
        "output_size": (8, 9),
        "kernel_size": (6, 6),
        "dilation": 1,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Fold"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Fold_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Fold_1'.")


check_valid('torch.nn.Fold', generated_inputs['torch.nn.Fold_1'], lib="torch", suffix=1)
