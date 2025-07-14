
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = 2
    stride = 1
    padding = 0
    ceil_mode = False
    count_include_pad = True
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    ceil_mode = False
    count_include_pad = False
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    ceil_mode = True
    count_include_pad = False
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = 2
    stride = 2
    padding = 0
    ceil_mode = False
    count_include_pad = True
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]], dtype=np.float32)
    kernel_size = 4
    stride = 3
    padding = 1
    ceil_mode = True
    count_include_pad = True
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]], dtype=np.float32)
    kernel_size = 4
    stride = 3
    padding = 1
    ceil_mode = True
    count_include_pad = True
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    kernel_size = 1
    stride = 1
    padding = 0
    ceil_mode = False
    count_include_pad = False
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Different input values
    input = np.array([[[5.0, 4.0, 3.0, 2.0, 1.0]]], dtype=np.float32)
    kernel_size = 2
    stride = 1
    padding = 0
    ceil_mode = False
    count_include_pad = True
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger padding
    input = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = 3
    stride = 1
    padding = 2
    ceil_mode = False
    count_include_pad = True
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-unit stride and padding
    input = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]], dtype=np.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    ceil_mode = False
    count_include_pad = False
    input_dict = {"input": input, "kernel_size": kernel_size, "stride": stride, "padding": padding, "ceil_mode": ceil_mode, "count_include_pad": count_include_pad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.avg_pool1d_1"] = avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_1'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_1'], lib="torch", suffix=1)
