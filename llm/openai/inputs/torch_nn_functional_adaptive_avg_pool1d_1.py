
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def adaptive_avg_pool1d_inputs():
    list_of_inputs = []

    input_arr = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8).numpy()
    input_dict = {"input": input_arr, "output_size": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 5, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "output_size": np.int32(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 2, 7, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "output_size": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.randn(1, 4, 3, dtype=torch.float32) * 5 - 2).numpy()
    input_dict = {"input": input_arr, "output_size": np.int64(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones(3, 1, 1, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(-10, 10, steps=5 * 5 * 10, dtype=torch.float32).reshape(5, 5, 10).numpy()
    input_dict = {"input": input_arr, "output_size": np.int32(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(2 * 2 * 9, dtype=torch.float32).reshape(2, 2, 9).numpy()
    input_dict = {"input": input_arr, "output_size": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.randn(1, 1, 50, dtype=torch.float64) * 1e6).numpy()
    input_dict = {"input": input_arr, "output_size": 25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.zeros(8, 1, 4, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 6, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "output_size": np.int64(7)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 3, 2, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "output_size": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(0, 1, steps=1 * 2 * 100, dtype=torch.float32).reshape(1, 2, 100).numpy()
    input_dict = {"input": input_arr, "output_size": 64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 5, dtype=torch.float32).numpy()
    input_arr[..., 2] = np.nan
    input_dict = {"input": input_arr, "output_size": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[[float('inf'), -float('inf'), 1.0, -1.0, 0.0, 2.0]]], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "output_size": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_avg_pool1d_1"] = adaptive_avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_avg_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_avg_pool1d_1'.")


check_valid('torch.nn.functional.adaptive_avg_pool1d', generated_inputs['torch.nn.functional.adaptive_avg_pool1d_1'], lib="torch", suffix=1)
