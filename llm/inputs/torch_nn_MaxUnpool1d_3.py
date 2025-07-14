
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def maxunpool1d_inputs():
    list_of_inputs = []

    # Input 1
    kernel_size = 2
    stride = (2,)
    padding = 0
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).numpy()
    indices_tensor = torch.tensor([[[0, 1]]]).type(torch.LongTensor).numpy()
    output_size = (1, 1, 8)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    kernel_size = 3
    stride = (1,)
    padding = 1
    input_tensor = torch.tensor([[[1.0, 2.0]]]).numpy()
    indices_tensor = torch.tensor([[[0, 1]]]).type(torch.LongTensor).numpy()
    output_size = (1, 1, 4)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    kernel_size = 2
    stride = None
    padding = 0
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0]]]).numpy()
    indices_tensor = torch.tensor([[[0, 2]]]).type(torch.LongTensor).numpy()
    output_size = (1, 1, 6)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": (2,),
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: No output_size
    kernel_size = 2
    stride = (2,)
    padding = 0
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).numpy()
    indices_tensor = torch.tensor([[[0, 2]]]).type(torch.LongTensor).numpy()
    output_size = (1, 1, 6)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: C, Hin format
    kernel_size = 2
    stride = (2,)
    padding = 0
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).numpy()
    indices_tensor = torch.tensor([[0, 2]]).type(torch.LongTensor).numpy()
    output_size = (1, 8)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different kernel_size and stride
    kernel_size = 3
    stride = (2,)
    padding = 0
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0]]]).numpy()
    indices_tensor = torch.tensor([[[0]]]).type(torch.LongTensor).numpy()
    output_size = (1, 1, 7)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Larger input size
    kernel_size = 2
    stride = (2,)
    padding = 0
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]).numpy()
    indices_tensor = torch.tensor([[[0, 2, 4, 6]]]).type(torch.LongTensor).numpy()
    output_size = (1, 1, 16)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_tensor,
        "indices": indices_tensor,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxUnpool1d_3"] = maxunpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_3'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_3'], lib="torch", suffix=3)
