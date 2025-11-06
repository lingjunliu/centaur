
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def adaptive_avg_pool2d_2_inputs():
    list_of_inputs = []

    input = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8).numpy()
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(2, 3, 7, 5, dtype=torch.float32).numpy()
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.linspace(-10, 10, steps=3 * 5 * 5, dtype=torch.float64).reshape(3, 5, 5).numpy()
    output_size = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (torch.randn(4, 2, 10, 10, dtype=torch.float32) * 10).numpy()
    output_size = 5
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.ones(1, 4, 6, dtype=torch.float32).mul(-2.5).numpy()
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.arange(1 * 1 * 9 * 4, dtype=torch.float64).reshape(1, 1, 9, 4).sub(50.0).numpy()
    output_size = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(3, 5, 3, 7, dtype=torch.float32).numpy()
    output_size = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.full((2, 9, 9), 1000.0, dtype=torch.float64).numpy()
    output_size = 9
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.zeros(1, 3, 1, 1, dtype=torch.float32).numpy()
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (torch.rand(2, 4, 6, 6, dtype=torch.float32) * 2 - 1).numpy()
    output_size = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.arange(5 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2).numpy()
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(1, 8, 32, 16, dtype=torch.float64).numpy()
    output_size = 8
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_avg_pool2d_2"] = adaptive_avg_pool2d_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_avg_pool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_avg_pool2d_2'.")


check_valid('torch.nn.functional.adaptive_avg_pool2d', generated_inputs['torch.nn.functional.adaptive_avg_pool2d_2'], lib="torch", suffix=2)
