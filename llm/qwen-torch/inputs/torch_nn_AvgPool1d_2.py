
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def avgpool1d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with kernel_size=3, stride=2
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7.]]]).numpy()
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - With padding
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8.]]]).numpy()
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (1,),
        "ceil_mode": False,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With ceil_mode=True
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8.]]]).numpy()
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": True,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With count_include_pad=False
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8.]]]).numpy()
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - Multi-dimensional input
    input = torch.tensor([[[[1., 2., 3., 4., 5., 6., 7., 8.]]], [[1., 2., 3., 4., 5., 6., 7., 8.]]]).numpy()
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Negative values in input
    input = torch.tensor([[[[-1., -2., -3., -4., -5., -6., -7.]]]).numpy()
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Large kernel size with small stride
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]]]).numpy()
    input_dict = {
        "kernel_size": (5,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Large stride with padding
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]]]).numpy()
    input_dict = {
        "kernel_size": (3,),
        "stride": (4,),
        "padding": (2,),
        "ceil_mode": False,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - With all parameters set to true
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]]]).numpy()
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (1,),
        "ceil_mode": True,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Mixed dimensions
    input = torch.tensor([[[[1., 2., 3., 4., 5., 6., 7., 8.]]], [[1., 2., 3., 4., 5., 6., 7., 8.]], [[1., 2., 3., 4., 5., 6., 7., 8.]]]).numpy()
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AvgPool1d_2"] = avgpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool1d_2'.")


check_valid('torch.nn.AvgPool1d', generated_inputs['torch.nn.AvgPool1d_2'], lib="torch", suffix=2)
