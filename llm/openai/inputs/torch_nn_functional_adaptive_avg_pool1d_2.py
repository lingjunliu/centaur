
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_avg_pool1d_inputs():
    list_of_inputs = []

    input = torch.arange(8, dtype=torch.float32).view(1, 1, 8).numpy()
    output_size = (1,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.linspace(-3, 3, steps=30, dtype=torch.float32).reshape(2, 3, 5).numpy()
    output_size = (2,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.tensor([[[1.0, -2.0, 3.0]],
                          [[-4.0, 5.0, -6.0]],
                          [[7.0, -8.0, 9.0]],
                          [[-10.0, 11.0, -12.0]]], dtype=torch.float32).numpy()
    output_size = (5,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(1, 4, 7, dtype=torch.float64).numpy()
    output_size = (7,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.asfortranarray(torch.randn(3, 2, 10, dtype=torch.float32).numpy())
    output_size = (4,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(5, 1, 100, dtype=torch.float32).numpy()
    output_size = (10,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.ones(1, 5, 1, dtype=torch.float32).numpy()
    output_size = (3,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (torch.randn(2, 4, 9, dtype=torch.float32) * 10 - 5).numpy()
    output_size = (4,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.zeros(2, 1, 16, dtype=torch.float32).numpy()
    output_size = (8,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(1, 2, 13, dtype=torch.float64).numpy()
    output_size = (6,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.linspace(0, 1, steps=12, dtype=torch.float32).reshape(1, 2, 6).numpy()
    output_size = (2,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    base = torch.randn(2, 3, 12, dtype=torch.float32).numpy()
    input = base[:, :, ::2]
    output_size = (3,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_avg_pool1d_2"] = adaptive_avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_avg_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_avg_pool1d_2'.")


check_valid('torch.nn.functional.adaptive_avg_pool1d', generated_inputs['torch.nn.functional.adaptive_avg_pool1d_2'], lib="torch", suffix=2)
