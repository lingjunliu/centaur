
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def split_inputs_1():
    list_of_inputs = []

    tensor = torch.arange(10, dtype=torch.float32).numpy()
    split_size_or_sections = 3
    dim = 0
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = torch.arange(10).reshape(5, 2).numpy()
    split_size_or_sections = 2
    dim = 0
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = torch.arange(10).reshape(5, 2).numpy()
    split_size_or_sections = 1
    dim = 1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = torch.randint(0, 2, (2, 3, 4), dtype=torch.int8).bool().numpy()
    split_size_or_sections = 2
    dim = -1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = torch.randn(7, 1, 5, dtype=torch.float64).numpy()
    split_size_or_sections = 4
    dim = 0
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = torch.arange(2*3*4*5, dtype=torch.int8).reshape(2, 3, 4, 5).numpy()
    split_size_or_sections = 3
    dim = 2
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = np.arange(5, dtype=np.int32)
    split_size_or_sections = 10
    dim = 0
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = torch.ones(1, 8, 2, 2, dtype=torch.float16).numpy()
    split_size_or_sections = 4
    dim = 1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = torch.randn(2, 2, 3, 3, 4, dtype=torch.float32).numpy()
    split_size_or_sections = 2
    dim = -3
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = torch.arange(27, dtype=torch.int64).reshape(3, 3, 3).numpy()
    split_size_or_sections = 1
    dim = 1
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    base = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    tensor = base[:, :, ::2]
    split_size_or_sections = 1
    dim = -2
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    tensor = torch.linspace(0, 1, steps=12, dtype=torch.float32).reshape(3, 4).T.numpy()
    split_size_or_sections = 5
    dim = 0
    list_of_inputs.append(copy.deepcopy({
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }))

    return list_of_inputs

generated_inputs["torch.split_1"] = split_inputs_1()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.split_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.split_1'.")


check_valid('torch.split', generated_inputs['torch.split_1'], lib="torch", suffix=1)
