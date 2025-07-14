
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv_transpose3d_inputs():
    list_of_inputs = []

    # Input 1
    input = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    weight = np.random.rand(3, 3, 3, 3, 3).astype(np.float32)
    bias = np.random.rand(3).astype(np.float32)
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

    # Input 2
    input = np.random.rand(2, 4, 5, 5, 5).astype(np.float32)
    weight = np.random.rand(4, 4, 2, 2, 2).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)
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

    # Input 3
    input = np.random.rand(1, 2, 8, 8, 8).astype(np.float32)
    weight = np.random.rand(2, 2, 4, 4, 4).astype(np.float32)
    bias = np.random.rand(2).astype(np.float32)
    stride = 1
    padding = 2
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

   # Input 4
    input = np.random.rand(1, 1, 4, 4, 4).astype(np.float32)
    weight = np.random.rand(1, 1, 1, 1, 1).astype(np.float32)
    bias = np.random.rand(1).astype(np.float32)
    stride = 3
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

    # Input 5
    input = np.random.rand(3, 5, 6, 6, 6).astype(np.float32)
    weight = np.random.rand(5, 5, 3, 3, 3).astype(np.float32)
    bias = np.random.rand(5).astype(np.float32)
    stride = 1
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

    # Input 6
    input = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    weight = np.random.rand(3, 3, 3, 3, 3).astype(np.float32) #Corrected Shape
    bias = np.random.rand(3).astype(np.float32)
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1 # Changed groups to 1 from 3
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

    # Input 7
    input = np.random.rand(1, 1, 7, 7, 7).astype(np.float32)
    weight = np.random.rand(1, 1, 2, 2, 2).astype(np.float32)
    bias = np.random.rand(1).astype(np.float32)
    stride = 2
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

    # Input 8
    input = np.random.rand(1, 2, 6, 6, 6).astype(np.float32)
    weight = np.random.rand(2, 2, 3, 3, 3).astype(np.float32)
    bias = np.random.rand(2).astype(np.float32)
    stride = 1
    padding = 1
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

    # Input 9
    input = np.random.rand(4, 3, 8, 8, 8).astype(np.float32)
    weight = np.random.rand(3, 3, 4, 4, 4).astype(np.float32)
    bias = np.random.rand(3).astype(np.float32)
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

    # Input 10
    input = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    weight = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
    bias = np.random.rand(1).astype(np.float32)
    stride = 1
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
generated_inputs["torch.nn.functional.conv_transpose3d"] = conv_transpose3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv_transpose3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv_transpose3d'.")

check_valid('torch.nn.functional.conv_transpose3d', generated_inputs['torch.nn.functional.conv_transpose3d'], lib="torch", suffix=0)
