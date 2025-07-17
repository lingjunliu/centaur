
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def max_unpool1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[1.0, 2.0, 3.0]])
    indices_tensor = np.array([[0, 1, 2]])
    kernel_size_val = 2
    stride_val = 1
    padding_val = 0
    output_size_val = (1, 6)

    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    indices_tensor = np.array([[[0, 1], [2, 3]]])
    kernel_size_val = 2
    stride_val = 2
    padding_val = 0
    output_size_val = (1, 4)
    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[ -1.0, -2.0, -3.0]]])
    indices_tensor = np.array([[[0, 2, 4]]])
    kernel_size_val = 3
    stride_val = 2
    padding_val = 1
    output_size_val = (1, 7)

    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1.0, 2.0]]])
    indices_tensor = np.array([[[0, 1]]])
    kernel_size_val = 1
    stride_val = 1
    padding_val = 0
    output_size_val = (1, 2)
    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1.0]]])
    indices_tensor = np.array([[[0]]])
    kernel_size_val = 3
    stride_val = 1
    padding_val = 1
    output_size_val = (1, 3)
    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    indices_tensor = np.array([[[0, 1, 2, 3]]])
    kernel_size_val = 4
    stride_val = 1
    padding_val = 0
    output_size_val = (1, 7)
    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[1.0]]])
    indices_tensor = np.array([[[1]]])
    kernel_size_val = 2
    stride_val = 2
    padding_val = 0
    output_size_val = (1, 4)

    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1.0, 2.0]]])
    indices_tensor = np.array([[[0, 2]]])
    kernel_size_val = 3
    stride_val = 1
    padding_val = 0
    output_size_val = (1, 5)

    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[1.0, 2.0]]])
    indices_tensor = np.array([[[2, 3]]])
    kernel_size_val = 3
    stride_val = 1
    padding_val = 0
    output_size_val = (1, 5)

    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[1.0]]])
    indices_tensor = np.array([[[2]]])
    kernel_size_val = 2
    stride_val = 1
    padding_val = 0
    output_size_val = (1, 3)

    input_dict = {
        "input": input_tensor,
        "indices": indices_tensor,
        "kernel_size": kernel_size_val,
        "stride": stride_val,
        "padding": padding_val,
        "output_size": output_size_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.max_unpool1d"] = max_unpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.max_unpool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_unpool1d'.")

check_valid('torch.nn.functional.max_unpool1d', generated_inputs['torch.nn.functional.max_unpool1d'], lib="torch", suffix=0)
