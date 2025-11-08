
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def adaptive_max_pool1d_inputs():
    list_of_inputs = []

    # 1
    input = torch.arange(8., dtype=torch.float32).reshape(1, 1, 8).numpy()
    output_size = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 2
    input = torch.randn(2, 3, 7, dtype=torch.float64).numpy()
    output_size = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 3 (no batch dim)
    input = torch.linspace(-5.0, 5.0, steps=9, dtype=torch.float32).repeat(4, 1).numpy()
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 4
    input = torch.zeros((5, 1, 1), dtype=torch.float32).numpy()
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 5
    input = torch.tensor([[[1.0, -2.0, 3.5, 0.0, -1.0],
                           [2.2, 2.2, -3.0, 4.0, 5.5]]], dtype=torch.float32).numpy()
    output_size = 5
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 6
    input = torch.linspace(-10, 10, steps=60, dtype=torch.float32).reshape(3, 2, 10).numpy()
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 7 (no batch dim)
    input = torch.tensor([[0.0, -1.0, -2.0, 3.0, 4.0, -5.0]], dtype=torch.float32).numpy()
    output_size = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 8 with NaNs
    x = torch.randn(2, 4, 13, dtype=torch.float32)
    x[0, 1, 5] = float('nan')
    x[1, 3, 7] = float('nan')
    input = x.numpy()
    output_size = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 9
    input = torch.sin(torch.linspace(0, 20, steps=50, dtype=torch.float32)).reshape(1, 1, 50).numpy()
    output_size = 10
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 10
    input = (torch.arange(36, dtype=torch.float64).reshape(4, 3, 3) - 18.0).numpy()
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 11
    input = torch.randn(1, 8, 16, dtype=torch.float32).numpy()
    output_size = 8
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    # 12
    input = torch.tensor([[[1000.0, -1000.0, 500.0, -500.0]],
                          [[-1e6, 1e6, -2e6, 2e6]]], dtype=torch.float32).repeat(1, 2, 1).numpy()
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool1d_1"] = adaptive_max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_max_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool1d_1'.")


check_valid('torch.nn.functional.adaptive_max_pool1d', generated_inputs['torch.nn.functional.adaptive_max_pool1d_1'], lib="torch", suffix=1)
