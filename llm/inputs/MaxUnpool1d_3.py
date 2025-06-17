
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MaxUnpool1d_inputs():
    list_of_inputs = []

    # Case 1: Basic case with N, C, Hin
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0]]])
    indices_tensor = torch.tensor([[[1, 1, 1]]])
    kernel_size = 2
    stride = 2
    padding = 0
    output_size = (1, 3, 7)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": (stride,),
        "padding": padding,
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: C, Hin
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    indices_tensor = torch.tensor([[1, 1, 1, 1]])
    kernel_size = 2
    stride = 1
    padding = 0
    input_dict = {
        "kernel_size": kernel_size,
        "stride": (stride,),
        "padding": padding,
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 3: No output_size, default stride
    input_tensor = torch.tensor([[[1.0, 2.0]]])
    indices_tensor = torch.tensor([[[1, 1]]])
    kernel_size = 2
    stride = None
    padding = 0
    input_dict = {
        "kernel_size": kernel_size,
        "stride": None,
        "padding": padding,
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different kernel_size, stride, padding values
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    indices_tensor = torch.tensor([[[1, 1, 1, 1]]])
    kernel_size = 3
    stride = 2
    padding = 1
    output_size = (1, 1, 9)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": (stride,),
        "padding": padding,
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Larger input, adjusted indices to fit output size constraint
    input_tensor = torch.randn(2, 3, 5)
    indices_tensor = torch.randint(0, 3, (2, 3, 3))
    kernel_size = 2
    stride = 2
    padding = 0
    output_size = (2, 3, 8)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": (stride,),
        "padding": padding,
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxUnpool1d_3"] = MaxUnpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_3'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_3'], lib="torch")
