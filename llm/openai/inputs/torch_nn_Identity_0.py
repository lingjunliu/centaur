
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def identity_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 vector
    args = []
    kwargs = []
    input_arr = torch.randn(5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 2: 2D float32 matrix
    args = [54]
    kwargs = [0]
    input_arr = torch.randn(8, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 3: scalar (0-d) float64
    args = [1, 2]
    kwargs = [0.1]
    input_arr = torch.tensor(3.14, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 4: 3D int32 tensor with negatives
    args = [0, 0]
    kwargs = []
    input_arr = torch.randint(-5, 6, (2, 3, 4), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 5: 4D bool tensor
    args = [1]
    kwargs = [1, 2, 3]
    input_arr = (torch.randn(2, 2, 2, 2) > 0).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 6: 5D float16 tensor
    args = [999, 0]
    kwargs = [0]
    input_arr = torch.randn(2, 1, 2, 2, 3, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 7: 2D float32 tensor
    args = [3]
    kwargs = [2]
    input_arr = torch.randn(3, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 8: empty (0, 4) float32 tensor
    args = [42]
    kwargs = [7]
    input_arr = torch.empty(0, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 9: values including NaN and Inf
    args = [0]
    kwargs = [1]
    input_arr = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 10: non-contiguous via transpose (int64)
    args = [3, 4]
    kwargs = [0]
    t = torch.arange(12, dtype=torch.int64).reshape(3, 4).t()
    input_arr = t.numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 11: larger dimensional float64 tensor
    args = [0, 0, 0]
    kwargs = [5]
    input_arr = torch.randn(2, 3, 4, 2, 2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 12: 1D empty vector
    args = [0]
    kwargs = []
    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 13: unsigned int8 tensor
    args = [255]
    kwargs = [8]
    input_arr = torch.randint(0, 256, (3, 3), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    # Input 14: 3D float32 tensor with zeros
    args = [0]
    kwargs = [1]
    input_arr = torch.zeros(2, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"args": args, "kwargs": kwargs, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.Identity"] = identity_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Identity'.")


check_valid('torch.nn.Identity', generated_inputs['torch.nn.Identity'], lib="torch", suffix=0)
