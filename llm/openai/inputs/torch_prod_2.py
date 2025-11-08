
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def prod_inputs_for_torch_prod_2():
    list_of_inputs = []

    # 1
    input = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    dim = 0
    keepdim = False
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 2
    input = torch.tensor([[1, 2, 0], [-1, 4, -3]], dtype=torch.int64).numpy()
    dim = 1
    keepdim = False
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 3
    input = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    dim = -1
    keepdim = True
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 4
    input = torch.arange(2 * 3 * 1 * 4, dtype=torch.int32).reshape(2, 3, 1, 4).numpy()
    dim = 2
    keepdim = False
    dtype = torch.int32
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 5
    input = torch.randn(2, 1, 3, 1, dtype=torch.float16).numpy()
    dim = 0
    keepdim = True
    dtype = torch.float16
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 6
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.uint8).numpy()
    dim = 2
    keepdim = False
    dtype = torch.uint8
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 7
    input = torch.full((3, 3), 100000, dtype=torch.int64).numpy()
    dim = 0
    keepdim = False
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 8
    input = torch.tensor([[-0.5, -1.5, 2.0, 3.0, -4.0]], dtype=torch.float64).numpy()
    dim = 1
    keepdim = True
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 9
    input = torch.randn(2, 2, 2, 2, 2, dtype=torch.float32).numpy()
    dim = 3
    keepdim = False
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 10
    input = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5).numpy()
    dim = -3
    keepdim = True
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 11
    input = torch.tensor([2, -3, 4, -5, 6, -7], dtype=torch.int16).numpy()
    dim = 0
    keepdim = True
    dtype = torch.int16
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    # 12 (complex)
    real = torch.randn(3, 3, dtype=torch.float32)
    imag = torch.randn(3, 3, dtype=torch.float32)
    input = (real + 1j * imag).numpy()
    dim = 0
    keepdim = False
    dtype = torch.complex128
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}))

    return list_of_inputs

generated_inputs["torch.prod_2"] = prod_inputs_for_torch_prod_2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.prod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.prod_2'.")


check_valid('torch.prod', generated_inputs['torch.prod_2'], lib="torch", suffix=2)
