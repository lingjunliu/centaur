
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_split_2_inputs():
    list_of_inputs = []

    # Input 1
    tensor = torch.arange(7).numpy()
    split_size_or_sections = [2, 3, 2]
    dim = 0
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 2
    tensor = torch.arange(10, dtype=torch.float32).reshape(5, 2).numpy()
    split_size_or_sections = [1, 4]
    dim = 0
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 3
    tensor = torch.randn(4, 6, dtype=torch.float32).numpy()
    split_size_or_sections = [2, 2, 2]
    dim = 1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 4
    tensor = torch.arange(3 * 4 * 5, dtype=torch.int32).reshape(3, 4, 5).numpy()
    split_size_or_sections = [1, 4]
    dim = -1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 5
    tensor = (torch.rand(6, 5, 4) > 0.5).numpy()
    split_size_or_sections = [2, 1, 2]
    dim = 1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 6
    tensor = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    split_size_or_sections = [1, 3]
    dim = -2
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 7
    tensor = torch.randn(7, 3, 3, 3, dtype=torch.float64).numpy()
    split_size_or_sections = [3, 4]
    dim = -4
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 8
    tensor = torch.linspace(0, 1, steps=8, dtype=torch.float64).numpy()
    split_size_or_sections = [3, 3, 2]
    dim = 0
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 9
    tensor = torch.arange(-20, 0, dtype=torch.int64).reshape(2, 10).numpy()
    split_size_or_sections = [1, 2, 1, 3, 3]
    dim = 1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 10
    tensor = torch.arange(2 * 2 * 2 * 2 * 6, dtype=torch.float16).reshape(2, 2, 2, 2, 6).numpy()
    split_size_or_sections = [1, 1, 2, 2]
    dim = -1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 11
    tensor = torch.randint(0, 256, (3, 5, 6), dtype=torch.uint8).numpy()
    split_size_or_sections = [2, 3]
    dim = 1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    # Input 12
    tensor = torch.randn(3, 4, dtype=torch.complex128).numpy()
    split_size_or_sections = [1, 3]
    dim = -1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    return list_of_inputs

generated_inputs["torch.split_2"] = torch_split_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.split_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.split_2'.")


check_valid('torch.split', generated_inputs['torch.split_2'], lib="torch", suffix=2)
