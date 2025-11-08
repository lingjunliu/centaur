
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_prod_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, 2.0, 3.5], dtype=torch.float32).numpy()
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.tensor([[0.5, -2.0], [3.0, 4.0]], dtype=torch.float64).numpy()
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.tensor([[[2, 3], [4, 5]], [[-6, 7], [8, -9]]], dtype=torch.int32).numpy()
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.arange(1, 13, dtype=torch.uint8).reshape(3, 2, 2).numpy()
    dtype = torch.int32
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.tensor([[-1.2, -0.5, 2.3], [0.0, -3.4, 1.1]], dtype=torch.float32).numpy()
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.tensor(7, dtype=torch.int64).numpy()
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.tensor([1+2j, -3+4j, -1j], dtype=torch.complex64).numpy()
    dtype = torch.complex64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.tensor([[1-1j, 2+0j], [3+5j, -2-2j]], dtype=torch.complex128).numpy()
    dtype = torch.complex128
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.tensor([[1.5, 2.0, -4.0], [7.5, -0.5, 0.25]], dtype=torch.float16).numpy()
    dtype = torch.float16
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.tensor([-5, -2, -3, 4], dtype=torch.int8).numpy()
    dtype = torch.int16
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.tensor([100000, 200000, 3000], dtype=torch.int64).numpy()
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    input = torch.arange(1, 25, dtype=torch.float32).reshape(1, 2, 3, 4).numpy()
    dtype = None
    list_of_inputs.append(copy.deepcopy({"input": input, "dtype": dtype}))

    return list_of_inputs

generated_inputs["torch.prod_1"] = torch_prod_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.prod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.prod_1'.")


check_valid('torch.prod', generated_inputs['torch.prod_1'], lib="torch", suffix=1)
