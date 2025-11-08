
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def conv_transpose2d_inputs():
    list_of_inputs = []

    # Input 1
    N, C_in, H, W = 1, 3, 5, 5
    out_channels, groups = 4, 1
    kH, kW = 3, 3
    input = torch.randn(N, C_in, H, W, dtype=torch.float32).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float32).numpy()
    bias = torch.randn(out_channels, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (1, 1),
        "padding": (0, 0),
        "output_padding": (0, 0),
        "groups": groups,
        "dilation": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    N, C_in, H, W = 2, 2, 8, 8
    out_channels, groups = 6, 1
    kH, kW = 2, 2
    input = torch.randn(N, C_in, H, W, dtype=torch.float64).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float64).numpy()
    bias = torch.randn(out_channels, dtype=torch.float64).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (2, 2),
        "padding": (1, 1),
        "output_padding": (1, 1),
        "groups": groups,
        "dilation": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    N, C_in, H, W = 1, 1, 4, 3
    out_channels, groups = 1, 1
    kH, kW = 2, 3
    input = torch.randn(N, C_in, H, W, dtype=torch.float32).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float32).numpy()
    bias = torch.randn(out_channels, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (2, 1),
        "padding": (1, 2),
        "output_padding": (1, 0),
        "groups": groups,
        "dilation": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (groups=2)
    N, C_in, H, W = 1, 4, 7, 6
    out_channels, groups = 6, 2
    kH, kW = 3, 3
    input = torch.randn(N, C_in, H, W, dtype=torch.float32).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float32).numpy()
    bias = torch.randn(out_channels, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (1, 1),
        "padding": (1, 1),
        "output_padding": (0, 0),
        "groups": groups,
        "dilation": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (depthwise-like, groups=in_channels)
    N, C_in, H, W = 2, 6, 8, 9
    out_channels, groups = 6, 6
    kH, kW = 1, 1
    input = torch.randn(N, C_in, H, W, dtype=torch.float16).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float16).numpy()
    bias = torch.randn(out_channels, dtype=torch.float16).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (1, 1),
        "padding": (0, 0),
        "output_padding": (0, 0),
        "groups": groups,
        "dilation": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (dilation > 1)
    N, C_in, H, W = 1, 5, 10, 10
    out_channels, groups = 10, 1
    kH, kW = 3, 3
    input = torch.randn(N, C_in, H, W, dtype=torch.float32).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float32).numpy()
    bias = torch.randn(out_channels, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (2, 2),
        "padding": (2, 2),
        "output_padding": (1, 1),
        "groups": groups,
        "dilation": (2, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (non-square stride/dilation)
    N, C_in, H, W = 3, 2, 5, 4
    out_channels, groups = 2, 1
    kH, kW = 2, 2
    input = torch.randn(N, C_in, H, W, dtype=torch.float64).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float64).numpy()
    bias = torch.randn(out_channels, dtype=torch.float64).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (3, 2),
        "padding": (1, 0),
        "output_padding": (2, 1),
        "groups": groups,
        "dilation": (1, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (groups=4, asymmetric kernel)
    N, C_in, H, W = 1, 8, 6, 5
    out_channels, groups = 12, 4
    kH, kW = 4, 1
    input = torch.randn(N, C_in, H, W, dtype=torch.float32).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float32).numpy()
    bias = torch.randn(out_channels, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (1, 2),
        "padding": (0, 1),
        "output_padding": (0, 1),
        "groups": groups,
        "dilation": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (larger kernel)
    N, C_in, H, W = 4, 1, 7, 7
    out_channels, groups = 2, 1
    kH, kW = 5, 5
    input = torch.randn(N, C_in, H, W, dtype=torch.float16).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float16).numpy()
    bias = torch.randn(out_channels, dtype=torch.float16).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (1, 1),
        "padding": (2, 2),
        "output_padding": (0, 0),
        "groups": groups,
        "dilation": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (small H/W, groups=3)
    N, C_in, H, W = 2, 3, 1, 2
    out_channels, groups = 3, 3
    kH, kW = 2, 2
    input = torch.randn(N, C_in, H, W, dtype=torch.float64).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float64).numpy()
    bias = torch.randn(out_channels, dtype=torch.float64).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (2, 3),
        "padding": (0, 1),
        "output_padding": (1, 0),
        "groups": groups,
        "dilation": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (groups=2, mixed dilation)
    N, C_in, H, W = 1, 4, 3, 3
    out_channels, groups = 8, 2
    kH, kW = 2, 3
    input = torch.randn(N, C_in, H, W, dtype=torch.float32).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float32).numpy()
    bias = torch.randn(out_channels, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (2, 1),
        "padding": (0, 1),
        "output_padding": (1, 0),
        "groups": groups,
        "dilation": (2, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (stride (1,3), output_padding (0,2))
    N, C_in, H, W = 2, 2, 5, 3
    out_channels, groups = 4, 1
    kH, kW = 3, 2
    input = torch.randn(N, C_in, H, W, dtype=torch.float32).numpy()
    weight = torch.randn(C_in, out_channels // groups, kH, kW, dtype=torch.float32).numpy()
    bias = torch.randn(out_channels, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias,
        "stride": (1, 3),
        "padding": (1, 0),
        "output_padding": (0, 2),
        "groups": groups,
        "dilation": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.conv_transpose2d_2"] = conv_transpose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.conv_transpose2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv_transpose2d_2'.")


check_valid('torch.nn.functional.conv_transpose2d', generated_inputs['torch.nn.functional.conv_transpose2d_2'], lib="torch", suffix=2)
