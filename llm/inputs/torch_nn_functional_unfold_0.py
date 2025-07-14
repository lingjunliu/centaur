
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def unfold_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 1, 32, 32).numpy()
    kernel_size = (3, 3)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(2, 3, 64, 64).numpy()
    kernel_size = (5, 5)
    dilation = 2
    padding = 1
    stride = 2
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(4, 1, 16, 16).numpy()
    kernel_size = (2, 2)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 3, 28, 28).numpy()
    kernel_size = (7, 7)
    dilation = 3
    padding = 2
    stride = 3
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(2, 1, 10, 10).numpy()
    kernel_size = (3, 3)
    dilation = 1
    padding = 1
    stride = 1
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(1, 1, 5, 5).numpy()
    kernel_size = (2, 2)
    dilation = 1
    padding = 0
    stride = 2
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(3, 3, 20, 20).numpy()
    kernel_size = (4, 4)
    dilation = 2
    padding = 1
    stride = 4
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(1, 1, 40, 40).numpy()
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = torch.randn(2, 3, 15, 15).numpy()
    kernel_size = (3, 3)
    dilation = 1
    padding = 2
    stride = 1
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(4, 1, 30, 30).numpy()
    kernel_size = (5, 5)
    dilation = 1
    padding = 0
    stride = 3
    input_dict = {"input": input_tensor, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.unfold"] = unfold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.unfold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.unfold'.")

check_valid('torch.nn.functional.unfold', generated_inputs['torch.nn.functional.unfold'], lib="torch", suffix=0)
