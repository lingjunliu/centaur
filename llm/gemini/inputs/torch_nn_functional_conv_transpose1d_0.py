
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv_transpose1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.randn(1, 3, 10).astype(np.float32)
    weight_tensor = np.random.randn(3, 5, 2).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    stride_val = 1
    padding_val = 0
    output_padding_val = 0
    groups_val = 1
    dilation_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.randn(2, 4, 15).astype(np.float32)
    weight_tensor = np.random.randn(4, 6, 3).astype(np.float32)
    bias_tensor = np.random.randn(6).astype(np.float32)
    stride_val = 2
    padding_val = 1
    output_padding_val = 1
    groups_val = 1
    dilation_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randn(1, 2, 5).astype(np.float32)
    weight_tensor = np.random.randn(2, 4, 1).astype(np.float32)
    bias_tensor = np.zeros(4).astype(np.float32)
    stride_val = 1
    padding_val = 0
    output_padding_val = 0
    groups_val = 1
    dilation_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: groups > 1
    input_tensor = np.random.randn(1, 4, 8).astype(np.float32)
    weight_tensor = np.random.randn(4, 2, 3).astype(np.float32)
    bias_tensor = np.random.randn(2).astype(np.float32)
    stride_val = 1
    padding_val = 0
    output_padding_val = 0
    groups_val = 2
    dilation_val = 1
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: dilation > 1
    input_tensor = np.random.randn(1, 3, 7).astype(np.float32)
    weight_tensor = np.random.randn(3, 5, 2).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    stride_val = 1
    padding_val = 0
    output_padding_val = 0
    groups_val = 1
    dilation_val = 2
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger stride and padding
    input_tensor = np.random.randn(1, 3, 20).astype(np.float32)
    weight_tensor = np.random.randn(3, 5, 3).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    stride_val = 3
    padding_val = 2
    output_padding_val = 1
    groups_val = 1
    dilation_val = 1
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.randn(3, 2, 8).astype(np.float32)
    weight_tensor = np.random.randn(2, 5, 3).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    stride_val = 2
    padding_val = 1
    output_padding_val = 0
    groups_val = 1
    dilation_val = 1
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: No bias
    input_tensor = np.random.randn(1, 3, 12).astype(np.float32)
    weight_tensor = np.random.randn(3, 4, 2).astype(np.float32)
    bias_tensor = np.zeros(4).astype(np.float32)
    stride_val = 1
    padding_val = 0
    output_padding_val = 0
    groups_val = 1
    dilation_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: output_padding = stride - 1
    input_tensor = np.random.randn(1, 2, 7).astype(np.float32)
    weight_tensor = np.random.randn(2, 4, 3).astype(np.float32)
    bias_tensor = np.random.randn(4).astype(np.float32)
    stride_val = 2
    padding_val = 1
    output_padding_val = 1
    groups_val = 1
    dilation_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Smaller input size
    input_tensor = np.random.randn(1, 1, 3).astype(np.float32)
    weight_tensor = np.random.randn(1, 2, 2).astype(np.float32)
    bias_tensor = np.random.randn(2).astype(np.float32)
    stride_val = 1
    padding_val = 0
    output_padding_val = 0
    groups_val = 1
    dilation_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Even smaller
    input_tensor = np.random.randn(1, 2, 2).astype(np.float32)
    weight_tensor = np.random.randn(2, 3, 1).astype(np.float32)
    bias_tensor = np.random.randn(3).astype(np.float32) # Corrected Bias Size
    stride_val = 1
    padding_val = 0
    output_padding_val = 0
    groups_val = 1
    dilation_val = 1
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv_transpose1d"] = conv_transpose1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv_transpose1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv_transpose1d'.")

check_valid('torch.nn.functional.conv_transpose1d', generated_inputs['torch.nn.functional.conv_transpose1d'], lib="torch", suffix=0)
