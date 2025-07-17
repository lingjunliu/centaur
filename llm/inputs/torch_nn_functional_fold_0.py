
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fold_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    output_size = (2, 2)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]])
    output_size = (3, 3)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]])
    output_size = (3, 3)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    output_size = (2, 3)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]])
    output_size = (4, 4)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]])
    output_size = (3, 4)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    output_size = (2, 2)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]])
    output_size = (3, 3)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    output_size = (2, 3)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]])
    output_size = (4, 4)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    output_size = (2, 3)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]])
    output_size = (4, 4)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 2
    input_dict = {"input": input_tensor, "output_size": output_size, "kernel_size": kernel_size, "dilation": dilation, "padding": padding, "stride": stride}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.fold"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.fold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.fold'.")

check_valid('torch.nn.functional.fold', generated_inputs['torch.nn.functional.fold'], lib="torch", suffix=0)
