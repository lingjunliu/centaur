
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def adaptive_max_pool3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn((1, 1, 4, 4, 4), dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (2, 2, 2),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn((2, 3, 7, 5, 9), dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (3, 2, 5),
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.randn((3, 8, 9, 10), dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (5, 7, 9),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = (torch.randn((1, 2, 6, 4, 3), dtype=torch.float32) - 2.0).numpy()
    input_dict = {
        "output_size": (3, 2, 2),
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.randn((1, 1, 4, 5, 6), dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (2, 4, 3),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = torch.randn((1, 5, 5, 5), dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (1, 1, 1),
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = torch.randn((4, 2, 16, 8, 8), dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (8, 4, 4),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.randn((2, 1, 5, 7, 9), dtype=torch.float64).numpy()
    input_dict = {
        "output_size": (5, 3, 3),
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.randn((2, 1, 3, 4), dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (1, 2, 2),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = torch.linspace(-5, 5, steps=24, dtype=torch.float32).reshape(1, 1, 2, 3, 4).numpy()
    input_dict = {
        "output_size": (2, 3, 2),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.zeros((1, 5, 5, 5, 5), dtype=torch.float32).numpy()
    input_dict = {
        "output_size": (5, 5, 5),
        "return_indices": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input_arr = (-torch.rand((3, 3, 6, 6, 6), dtype=torch.float32)).numpy()
    input_dict = {
        "output_size": (3, 3, 3),
        "return_indices": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveMaxPool3d_2"] = adaptive_max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveMaxPool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveMaxPool3d_2'.")


check_valid('torch.nn.AdaptiveMaxPool3d', generated_inputs['torch.nn.AdaptiveMaxPool3d_2'], lib="torch", suffix=2)
