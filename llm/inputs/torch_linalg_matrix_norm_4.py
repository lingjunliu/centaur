
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_matrix_norm_inputs():
    list_of_inputs = []

    # The issue stems from 'ord' needing to satisfy two conflicting conditions based on the error history:
    # 1. Have a .shape attribute for the test harness (since the signature type is 'tensor').
    # 2. Be interpreted as a Number by PyTorch, not a Tensor.
    # The solution is to use numpy scalar types (e.g., np.int64, np.float64), which have a .shape
    # attribute but are treated as numbers by PyTorch, satisfying both conditions.
    # String 'ord' values like 'fro' are avoided as they don't have a .shape and would fail the harness
    # given the strict signature requirement {'ord': 'tensor'}.

    base_dict = { "dim": (0, 1), "keepdim": False, "out": None, "dtype": None }

    # Input 1: 2-norm, float32
    case1 = copy.deepcopy(base_dict)
    case1.update({
        "input": np.random.rand(3, 4).astype(np.float32),
        "ord": np.int64(2),
    })
    list_of_inputs.append(case1)

    # Input 2: 1-norm, float64, keepdim=True
    case2 = copy.deepcopy(base_dict)
    case2.update({
        "input": np.random.rand(5, 5).astype(np.float64),
        "ord": np.int64(1),
        "dim": (-2, -1),
        "keepdim": True,
    })
    list_of_inputs.append(case2)

    # Input 3: Infinity-norm, batched input
    case3 = copy.deepcopy(base_dict)
    case3.update({
        "input": np.random.rand(3, 2, 4).astype(np.float32),
        "ord": np.float64(np.inf),
        "dim": (1, 2),
    })
    list_of_inputs.append(case3)

    # Input 4: -1-norm, batched input, keepdim=True
    case4 = copy.deepcopy(base_dict)
    case4.update({
        "input": (np.random.rand(2, 4, 3) - 0.5).astype(np.float32),
        "ord": np.int64(-1),
        "dim": (1, 2),
        "keepdim": True,
    })
    list_of_inputs.append(case4)

    # Input 5: -2-norm
    case5 = copy.deepcopy(base_dict)
    case5.update({
        "input": np.random.rand(6, 2).astype(np.float32),
        "ord": np.int64(-2),
    })
    list_of_inputs.append(case5)

    # Input 6: -Infinity-norm
    case6 = copy.deepcopy(base_dict)
    case6.update({
        "input": np.random.rand(2, 2).astype(np.float32),
        "ord": np.float64(-np.inf),
        "keepdim": True,
    })
    list_of_inputs.append(case6)

    # Input 7: With 'out' tensor
    case7 = copy.deepcopy(base_dict)
    case7.update({
        "input": np.random.rand(4, 2, 3).astype(np.float32),
        "ord": np.int64(1),
        "dim": (1, 2),
        "out": np.zeros(4, dtype=np.float32),
    })
    list_of_inputs.append(case7)

    # Input 8: With 'dtype' specified
    case8 = copy.deepcopy(base_dict)
    case8.update({
        "input": np.random.rand(3, 5).astype(np.float32),
        "ord": np.int64(2),
        "dtype": np.float64,
    })
    list_of_inputs.append(case8)

    # Input 9: Complex input, 2-norm
    case9 = copy.deepcopy(base_dict)
    case9.update({
        "input": (np.random.rand(4, 4) + 1j * np.random.rand(4, 4)).astype(np.complex64),
        "ord": np.int64(2),
    })
    list_of_inputs.append(case9)

    # Input 10: Complex input, inf norm, with dtype specified for float output
    case10 = copy.deepcopy(base_dict)
    case10.update({
        "input": (np.random.rand(2, 3) + 1j * np.random.rand(2, 3)).astype(np.complex128),
        "ord": np.float64(np.inf),
        "keepdim": True,
        "dtype": np.float64,
    })
    list_of_inputs.append(case10)

    # Input 11: Higher dimensional input (4D)
    case11 = copy.deepcopy(base_dict)
    case11.update({
        "input": np.random.rand(2, 3, 4, 5).astype(np.float32),
        "ord": np.int64(2),
        "dim": (2, 3),
    })
    list_of_inputs.append(case11)

    return list_of_inputs

generated_inputs["torch.linalg.matrix_norm_4"] = linalg_matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.matrix_norm_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_norm_4'.")

check_valid('torch.linalg.matrix_norm', generated_inputs['torch.linalg.matrix_norm_4'], lib="torch", suffix=4)
