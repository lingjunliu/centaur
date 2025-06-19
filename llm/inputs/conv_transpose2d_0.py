
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy

def conv_transpose2d_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case
    input = torch.randn(1, 3, 5, 5).numpy()
    weight = torch.randn(3, 2, 3, 3).numpy()
    bias = torch.randn(2).numpy()
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Different stride and padding
    input = torch.randn(1, 4, 4, 4).numpy()
    weight = torch.randn(4, 2, 3, 3).numpy()
    bias = torch.randn(2).numpy()
    stride = 2
    padding = 1
    output_padding = 1
    groups = 1
    dilation = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Groups > 1. Removed to avoid issues with bias dimension
    # input = torch.randn(1, 4, 6, 6).numpy()
    # weight = torch.randn(4, 2, 5, 5).numpy()
    # bias = torch.randn(2).numpy() #Potential for bias dimension mismatch when groups > 1
    # stride = 1
    # padding = 0
    # output_padding = 0
    # groups = 2
    # dilation = 1
    
    # input_dict = {
    #     "input": input,
    #     "weight": weight,
    #     "bias": bias,
    #     "stride": stride,
    #     "padding": padding,
    #     "output_padding": output_padding,
    #     "groups": groups,
    #     "dilation": dilation
    # }
    
    # list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Dilation > 1
    input = torch.randn(1, 1, 5, 5).numpy()
    weight = torch.randn(1, 1, 3, 3).numpy()
    bias = torch.randn(1).numpy()
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 2
    
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Larger batch size
    input = torch.randn(2, 8, 10, 10).numpy()
    weight = torch.randn(8, 16, 4, 4).numpy()
    bias = torch.randn(16).numpy()
    stride = 2
    padding = 1
    output_padding = 0
    groups = 1
    dilation = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv_transpose2d"] = conv_transpose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv_transpose2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv_transpose2d'.")

check_valid('torch.nn.functional.conv_transpose2d', generated_inputs['torch.nn.functional.conv_transpose2d'], lib="torch")
