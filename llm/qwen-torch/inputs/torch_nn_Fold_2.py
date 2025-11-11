
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def fold_inputs():
    list_of_inputs = []
    
    # Input 1 - 4D input (batched) with proper shape
    input = torch.randn(1, 3 * 2 * 2, 12).numpy()
    input_dict = {
        "output_size": 4,
        "kernel_size": 2,
        "dilation": 1,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - 4D input (batched) with different output size
    input = torch.randn(1, 4 * 3 * 3, 16).numpy()
    input_dict = {
        "output_size": 5,
        "kernel_size": 3,
        "dilation": 1,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - 4D input with padding
    input = torch.randn(1, 2 * 2 * 2, 8).numpy()
    input_dict = {
        "output_size": 3,
        "kernel_size": 2,
        "dilation": 1,
        "padding": 1,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - 4D input with stride
    input = torch.randn(1, 3 * 3 * 3, 27).numpy()
    input_dict = {
        "output_size": 5,
        "kernel_size": 3,
        "dilation": 1,
        "padding": 0,
        "stride": 2,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - 4D input with dilation
    input = torch.randn(1, 3 * 3 * 3, 27).numpy()
    input_dict = {
        "output_size": 5,
        "kernel_size": 3,
        "dilation": 2,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Multi-dimensional output size
    input = torch.randn(1, 2 * 3 * 4, 24).numpy()
    input_dict = {
        "output_size": (3, 4),
        "kernel_size": (2, 3),
        "dilation": 1,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Large kernel size with small output size
    input = torch.randn(1, 2 * 2 * 2, 8).numpy()
    input_dict = {
        "output_size": 3,
        "kernel_size": 4,
        "dilation": 1,
        "padding": 0,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Large stride with small kernel size
    input = torch.randn(1, 3 * 3 * 3, 27).numpy()
    input_dict = {
        "output_size": 5,
        "kernel_size": 3,
        "dilation": 1,
        "padding": 0,
        "stride": 3,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Negative padding (should work)
    input = torch.randn(1, 2 * 2 * 2, 8).numpy()
    input_dict = {
        "output_size": 3,
        "kernel_size": 2,
        "dilation": 1,
        "padding": -1,
        "stride": 1,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With different dimensions (properly calculated)
    input = torch.randn(1, 3 * 3 * 3, 27).numpy()
    input_dict = {
        "output_size": 5,
        "kernel_size": 3,
        "dilation": 1,
        "padding": 0,
        "stride": 1,
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
