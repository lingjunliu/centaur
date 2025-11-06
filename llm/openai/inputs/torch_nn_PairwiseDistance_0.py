
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pairwise_distance_inputs():
    list_of_inputs = []

    # Input 1: 1D vectors, default-like
    input1 = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    input2 = torch.tensor([-1.0, 2.5, 0.5], dtype=torch.float32).numpy()
    p = 2.0
    eps = 1e-6
    keepdim = False
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 2: 2D batch, Manhattan distance
    input1 = torch.randn(4, 5, dtype=torch.float32).numpy()
    input2 = torch.randn(4, 5, dtype=torch.float32).numpy()
    p = 1.0
    eps = 1e-6
    keepdim = False
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 3: 2D batch, keepdim True
    input1 = torch.randn(3, 1, dtype=torch.float32).numpy()
    input2 = torch.randn(3, 1, dtype=torch.float32).numpy()
    p = 2.0
    eps = 1e-6
    keepdim = True
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 4: 1D vectors, negative p
    input1 = torch.tensor([0.5, -1.5, 2.0, -3.0], dtype=torch.float32).numpy()
    input2 = torch.tensor([-0.5, 1.0, -2.5, 3.5], dtype=torch.float32).numpy()
    p = -0.5
    eps = 1e-6
    keepdim = False
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 5: 2D batch, fractional p
    input1 = torch.randn(2, 6, dtype=torch.float32).numpy()
    input2 = torch.randn(2, 6, dtype=torch.float32).numpy()
    p = 1.5
    eps = 1e-8
    keepdim = False
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 6: 2D batch, large p, eps=0
    input1 = torch.randn(5, 3, dtype=torch.float32).numpy()
    input2 = torch.randn(5, 3, dtype=torch.float32).numpy()
    p = 10.0
    eps = 0.0
    keepdim = False
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 7: 2D batch with negative values, keepdim True
    input1 = torch.tensor([[-1.0, -2.0, -3.0],
                           [ 4.0, -5.0,  6.0]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[ 1.5, -2.5,  3.5],
                           [-4.5,  5.5, -6.5]], dtype=torch.float32).numpy()
    p = 3.0
    eps = 1e-6
    keepdim = True
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 8: 1D high-D vectors
    input1 = torch.randn(128, dtype=torch.float32).numpy()
    input2 = torch.randn(128, dtype=torch.float32).numpy()
    p = 2.0
    eps = 1e-6
    keepdim = False
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 9: 2D batch, float64 dtype
    input1 = torch.randn(3, 7, dtype=torch.float64).numpy()
    input2 = torch.randn(3, 7, dtype=torch.float64).numpy()
    p = 2.0
    eps = 1e-9
    keepdim = False
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 10: 2D non-contiguous via slicing
    base1 = torch.randn(6, 4, dtype=torch.float32).numpy()
    base2 = torch.randn(6, 4, dtype=torch.float32).numpy()
    input1 = base1[::2]  # shape (3, 4)
    input2 = base2[::2]  # shape (3, 4)
    p = 0.7
    eps = 1e-6
    keepdim = False
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 11: 1D vectors, negative p with larger magnitude
    input1 = torch.tensor([0.0, 0.0, 1.0, -1.0, 2.0], dtype=torch.float32).numpy()
    input2 = torch.tensor([1.0, -1.0, 0.0, 0.0, -2.0], dtype=torch.float32).numpy()
    p = -2.0
    eps = 1e-3
    keepdim = True
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    # Input 12: 2D batch with large magnitude numbers
    input1 = (torch.randn(2, 3, dtype=torch.float64) * 1e6).numpy()
    input2 = (torch.randn(2, 3, dtype=torch.float64) * 1e6).numpy()
    p = 2.0
    eps = 1e-12
    keepdim = True
    list_of_inputs.append(copy.deepcopy({
        "p": p, "eps": eps, "keepdim": keepdim,
        "input1": input1, "input2": input2
    }))

    return list_of_inputs

generated_inputs["torch.nn.PairwiseDistance"] = pairwise_distance_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.PairwiseDistance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.PairwiseDistance'.")


check_valid('torch.nn.PairwiseDistance', generated_inputs['torch.nn.PairwiseDistance'], lib="torch", suffix=0)
