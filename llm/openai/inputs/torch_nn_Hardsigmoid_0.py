
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hardsigmoid_inputs():
    list_of_inputs = []

    # Input 1: 1D small vector, float32
    input = torch.tensor([0.0], dtype=torch.float32).numpy()
    inplace = False
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 2: 1D with edge and extreme values, float64
    input = torch.tensor([-10.0, -3.0, -0.1, 0.0, 2.5, 3.0, 10.0], dtype=torch.float64).numpy()
    inplace = True
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 3: 2D matrix, random normal, float32
    input = torch.randn(2, 3, dtype=torch.float32).numpy()
    inplace = False
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 4: 3D tensor, float16
    input = torch.randn(2, 2, 2, dtype=torch.float16).numpy()
    inplace = True
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 5: 4D tensor, uniform values, float32
    input = torch.linspace(-5, 5, steps=60, dtype=torch.float32).view(1, 3, 4, 5).numpy()
    inplace = False
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 6: Empty 1D, float32
    input = torch.empty((0,), dtype=torch.float32).numpy()
    inplace = True
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 7: Contains NaN and Infs, float32
    input = torch.tensor([float('nan'), float('inf'), float('-inf'), -2.5, 2.5], dtype=torch.float32).numpy()
    inplace = False
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 8: Non-contiguous via transpose, float32
    input = torch.arange(12, dtype=torch.float32).view(3, 4).t().numpy()
    inplace = True
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 9: Very large magnitude values, float32
    input = torch.tensor([1e6, -1e6, 3.1, -3.1, 0.0], dtype=torch.float32).numpy()
    inplace = False
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 10: 5D tensor, float32
    input = torch.linspace(-4, 4, steps=24, dtype=torch.float32).view(2, 1, 3, 1, 4).numpy()
    inplace = True
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 11: 1D random negative values, float32
    input = (-5 * torch.rand(4, dtype=torch.float32)).numpy()
    inplace = False
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 12: 2D square matrix, float64
    input = torch.tensor([[-4.0, -2.0, 0.0],
                          [1.5, 3.5, 2.0],
                          [-3.0, 3.0, 0.5]], dtype=torch.float64).numpy()
    inplace = True
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    # Input 13: Empty 2D (batch with zero features), float32
    input = torch.empty((8, 0), dtype=torch.float32).numpy()
    inplace = False
    list_of_inputs.append(copy.deepcopy({"inplace": inplace, "input": input}))

    return list_of_inputs

generated_inputs["torch.nn.Hardsigmoid"] = hardsigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Hardsigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Hardsigmoid'.")


check_valid('torch.nn.Hardsigmoid', generated_inputs['torch.nn.Hardsigmoid'], lib="torch", suffix=0)
