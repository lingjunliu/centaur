
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool1d_inputs_case2():
    list_of_inputs = []

    input = np.ones((1, 1, 8), dtype=np.float32)
    output_size = (4,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randn(2, 3, 15).astype(np.float64)
    output_size = (5,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (np.random.randn(1, 2, 7).astype(np.float32) - 5.0)
    output_size = (1,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.arange(4 * 1 * 9, dtype=np.float32).reshape(4, 1, 9)
    output_size = (3,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randn(1, 4, 10).astype(np.float32)
    input[0, 0, 0] = np.nan
    input[0, 1, 1] = np.inf
    input[0, 2, 2] = -np.inf
    output_size = (10,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.randn(3, 2, 33).astype(np.float64)
    output_size = (11,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.array([[[1.0]], [[-3.0]]], dtype=np.float32)
    output_size = (1,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.linspace(-1e6, 1e6, num=1 * 5 * 17, dtype=np.float32).reshape(1, 5, 17)
    output_size = (2,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.linspace(-10, 10, num=5 * 3 * 6, dtype=np.float64).reshape(5, 3, 6)
    output_size = (6,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.random.rand(1, 2, 50).astype(np.float32)
    output_size = (25,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = (np.random.randn(8, 4, 13).astype(np.float32) * 100.0)
    output_size = (7,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    input = np.array([[[ -5.0, 0.0, 2.0 ]]], dtype=np.float32)
    output_size = (2,)
    list_of_inputs.append(copy.deepcopy({"input": input, "output_size": output_size}))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool1d_2"] = adaptive_max_pool1d_inputs_case2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_max_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool1d_2'.")


check_valid('torch.nn.functional.adaptive_max_pool1d', generated_inputs['torch.nn.functional.adaptive_max_pool1d_2'], lib="torch", suffix=2)
