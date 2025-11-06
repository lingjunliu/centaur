
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool3d_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(1, 1, 7, 3, 5, dtype=torch.float64).numpy()
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(1, 2, 2, 3, 4, dtype=torch.float32).numpy()
    output_size = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(3, 4, 5, 6, dtype=torch.float32).numpy()
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(2, 3, 4, 4, dtype=torch.float64).numpy()
    output_size = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (torch.randn(1, 3, 8, 8, 8, dtype=torch.float32) * 10 - 5).numpy()
    output_size = 5
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2).numpy()
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    base = torch.randn(2, 3, 7, 5, 9, dtype=torch.float32).numpy()
    input = np.transpose(base, (0, 1, 4, 2, 3))
    output_size = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (torch.randn(4, 1, 3, 3, 3, dtype=torch.float64) - 100.0).numpy()
    output_size = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.full((1, 2, 3, 3, 3), fill_value=-np.inf, dtype=torch.float32).numpy()
    input[0, 1, 2, 2, 2] = np.inf
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(1, 1, 1, 1, 10, dtype=torch.float32).numpy()
    output_size = 7
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = torch.randn(2, 2, 3, 3, 3, dtype=torch.float64).numpy()
    output_size = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool3d_2"] = adaptive_max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_max_pool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool3d_2'.")


check_valid('torch.nn.functional.adaptive_max_pool3d', generated_inputs['torch.nn.functional.adaptive_max_pool3d_2'], lib="torch", suffix=2)
