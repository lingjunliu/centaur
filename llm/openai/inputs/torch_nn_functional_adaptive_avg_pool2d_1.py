
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def adaptive_avg_pool2d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(2, 3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": (4, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(1, 1, 5, 7, dtype=torch.float64).numpy() * -3.5
    input_dict = {"input": input_arr, "output_size": (1, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.randn(3, 10, 3, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": (2, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.randn(4, 2, 9, 9, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "output_size": (3, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.randn(2, 4, 6, 9, dtype=torch.float32).numpy()
    input_arr[0, 0, 0, 0] = np.nan
    input_arr[1, 3, 5, 8] = np.inf
    input_dict = {"input": input_arr, "output_size": (2, 7)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = (torch.randn(5, 2, 2, dtype=torch.float64) * 10).numpy()
    input_dict = {"input": input_arr, "output_size": (5, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = (torch.randn(3, 3, 7, 4, dtype=torch.float32) * -2.0).numpy()
    input_dict = {"input": input_arr, "output_size": (3, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.randn(4, 6, 5, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": (6, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.randn(2, 16, 11, 13, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": (7, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = torch.randn(1, 2, 1, 10, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": (3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.randn(1, 1, 7, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "output_size": (1, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input_arr = torch.randn(5, 1, 3, 3, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "output_size": (2, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_avg_pool2d_1"] = adaptive_avg_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_avg_pool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_avg_pool2d_1'.")


check_valid('torch.nn.functional.adaptive_avg_pool2d', generated_inputs['torch.nn.functional.adaptive_avg_pool2d_1'], lib="torch", suffix=1)
