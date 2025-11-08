
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def adaptive_max_pool3d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(1, 1, 4, 6, 8, dtype=torch.float32).numpy()
    output_size = (1, 1, 1)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.randn(2, 3, 5, 5, 5, dtype=torch.float64).numpy()
    output_size = (2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.randn(3, 2, 7, 9, dtype=torch.float32).numpy()
    output_size = (1, 7, 3)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.ones(4, 1, 2, 8, 6, dtype=torch.float32).numpy()
    output_size = (2, 4, 3)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.zeros(1, 1, 10, 10, 10, dtype=torch.float32).numpy()
    output_size = (5, 5, 5)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = (torch.randn(3, 2, 4, 4, 4, dtype=torch.float32) - 10.0).numpy()
    output_size = (4, 1, 2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.randn(2, 2, 1, 3, 7, dtype=torch.float32).numpy()
    output_size = (1, 1, 7)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.linspace(-1.0, 1.0, steps=1 * 1 * 11 * 13 * 17, dtype=torch.float32).reshape(1, 1, 11, 13, 17).numpy()
    output_size = (11, 13, 17)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.randn(2, 3, 2, 2, 2, dtype=torch.float32).numpy()
    output_size = (1, 2, 2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.randn(5, 3, 2, 4, dtype=torch.float32).numpy()
    output_size = (1, 2, 2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.randn(7, 9, 11, 13, dtype=torch.float64).numpy()
    output_size = (3, 5, 7)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    input_arr = torch.ones(4, 5, 6, 7, 8, dtype=torch.float32).numpy()
    output_size = (3, 2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "output_size": output_size}))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool3d_1"] = adaptive_max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_max_pool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool3d_1'.")


check_valid('torch.nn.functional.adaptive_max_pool3d', generated_inputs['torch.nn.functional.adaptive_max_pool3d_1'], lib="torch", suffix=1)
