
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def avgpool1d_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7.]]]).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8.]]]).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9.]]]).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]]]).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]]]).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 2,
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]]]).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.]]]).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.]]]).numpy()
    input_dict = {
        "kernel_size": 6,
        "stride": 3,
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]]]).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]]]).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AvgPool1d_1"] = avgpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool1d_1'.")


check_valid('torch.nn.AvgPool1d', generated_inputs['torch.nn.AvgPool1d_1'], lib="torch", suffix=1)
