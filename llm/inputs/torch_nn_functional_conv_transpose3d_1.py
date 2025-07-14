
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv_transpose3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(2, 3, 10, 10, 10).astype(np.float32)
    weight_tensor = np.random.rand(3, 5, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(5).astype(np.float32)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    groups = 1
    dilation = (1, 1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    weight_tensor = np.random.rand(1, 1, 2, 2, 2).astype(np.float32)
    bias_tensor = np.random.rand(1).astype(np.float32)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)
    groups = 1
    dilation = (1, 1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(4, 2, 7, 7, 7).astype(np.float32)
    weight_tensor = np.random.rand(2, 4, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(4).astype(np.float32)
    stride = (1, 2, 1)
    padding = (1, 0, 1)
    output_padding = (0, 1, 0)
    groups = 1
    dilation = (2, 1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 4, 8, 8, 8).astype(np.float32)
    weight_tensor = np.random.rand(4, 4, 1, 1, 1).astype(np.float32)
    bias_tensor = np.random.rand(4).astype(np.float32)
    stride = (3, 3, 3)
    padding = (0, 0, 0)
    output_padding = (1, 1, 1)
    groups = 1
    dilation = (1, 1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(2, 2, 6, 6, 6).astype(np.float32)
    weight_tensor = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)
    bias_tensor = np.random.rand(2).astype(np.float32)
    stride = (1, 1, 1)
    padding = (0, 1, 0)
    output_padding = (0, 0, 0)
    groups = 1
    dilation = (1, 2, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 2, 4, 4, 4).astype(np.float32)
    weight_tensor = np.random.rand(2, 2, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(2).astype(np.float32)
    stride = (2, 1, 2)
    padding = (1, 0, 1)
    output_padding = (1, 0, 1)
    groups = 1
    dilation = (1, 1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 (groups > 1)
    input_tensor = np.random.rand(1, 4, 8, 8, 8).astype(np.float32)
    weight_tensor = np.random.rand(4, 1, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(4).astype(np.float32)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    output_padding = (0, 0, 0)
    groups = 4
    dilation = (1, 1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(2, 3, 5, 5, 5).astype(np.float32)
    weight_tensor = np.random.rand(3, 5, 1, 1, 1).astype(np.float32)
    bias_tensor = np.random.rand(5).astype(np.float32)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    groups = 1
    dilation = (2, 2, 2)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (dilation > 1)
    input_tensor = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    weight_tensor = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(1).astype(np.float32)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    output_padding = (0, 0, 0)
    groups = 1
    dilation = (2, 2, 2)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (bias is None)
    input_tensor = np.random.rand(2, 3, 10, 10, 10).astype(np.float32)
    weight_tensor = np.random.rand(3, 5, 3, 3, 3).astype(np.float32)
    bias_tensor = None
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    groups = 1
    dilation = (1, 1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv_transpose3d_1"] = conv_transpose3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv_transpose3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv_transpose3d_1'.")

check_valid('torch.nn.functional.conv_transpose3d', generated_inputs['torch.nn.functional.conv_transpose3d_1'], lib="torch", suffix=1)
