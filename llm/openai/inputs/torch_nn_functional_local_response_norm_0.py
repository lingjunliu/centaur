
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def local_response_norm_inputs():
    list_of_inputs = []

    # Input 1
    n = 1 * 3 * 4 * 4
    input_arr = torch.linspace(-1.0, 1.0, steps=n, dtype=torch.float32).reshape(1, 3, 4, 4).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int32(5),
        "alpha": np.float32(1e-4),
        "beta": np.float32(0.75),
        "k": np.float32(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    n = 2 * 4 * 3 * 3
    input_arr = torch.linspace(-2.0, 2.0, steps=n, dtype=torch.double).reshape(2, 4, 3, 3).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int64(3),
        "alpha": np.float64(1e-3),
        "beta": np.float64(1.0),
        "k": np.float64(2.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = (torch.ones((1, 8, 1, 1), dtype=torch.float32) * 2.0).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int32(2),
        "alpha": np.float32(1e-2),
        "beta": np.float32(0.5),
        "k": np.float32(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.zeros((4, 1, 10, 10), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int32(1),
        "alpha": np.float32(0.0),
        "beta": np.float32(0.75),
        "k": np.float32(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    n = 3 * 16 * 7 * 5
    input_arr = torch.linspace(-0.5, 0.5, steps=n, dtype=torch.float64).reshape(3, 16, 7, 5).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int32(7),
        "alpha": np.float64(1e-4),
        "beta": np.float64(0.5),
        "k": np.float64(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    n = 2 * 6 * 2 * 4
    base = torch.arange(n, dtype=torch.float32)
    input_arr = torch.sin(base).reshape(2, 6, 2, 4).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int32(4),
        "alpha": np.float32(1e-1),
        "beta": np.float32(0.9),
        "k": np.float32(0.5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    n = 5 * 5 * 5 * 5
    base = torch.arange(n, dtype=torch.float32)
    input_arr = torch.cos(base / 10.0).reshape(5, 5, 5, 5).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int32(3),
        "alpha": np.float32(1e-5),
        "beta": np.float32(2.0),
        "k": np.float32(3.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    n = 1 * 2 * 8 * 8
    base = torch.arange(n, dtype=torch.float64)
    pattern = ((base % 2) * 2 - 1).reshape(1, 2, 8, 8).to(torch.float64)
    input_arr = pattern.numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int64(5),
        "alpha": np.float64(0.0015),
        "beta": np.float64(0.75),
        "k": np.float64(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    n = 3 * 7 * 4 * 4
    input_arr = torch.linspace(-3.0, 3.0, steps=n, dtype=torch.float32).reshape(3, 7, 4, 4).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int32(6),
        "alpha": np.float32(0.00025),
        "beta": np.float32(1.25),
        "k": np.float32(10.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    base_tensor = torch.tensor([
        [[[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]]],
        [[[-0.8, 0.9, -1.0, 1.1, -1.2, 1.3, -1.4]]]
    ], dtype=torch.float32)
    input_arr = base_tensor.repeat(1, 3, 1, 1).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int32(1),
        "alpha": np.float32(0.05),
        "beta": np.float32(0.25),
        "k": np.float32(0.1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    n = 1 * 32 * 2 * 2
    base = torch.arange(n, dtype=torch.float32)
    input_arr = torch.tanh(base / 5.0).reshape(1, 32, 2, 2).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int32(15),
        "alpha": np.float32(1e-4),
        "beta": np.float32(0.75),
        "k": np.float32(1.2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    n = 6 * 4 * 2 * 3
    base = torch.arange(n, dtype=torch.double)
    input_arr = (torch.sin(base) * torch.cos(base / 3.0)).reshape(6, 4, 2, 3).numpy()
    input_dict = {
        "input": input_arr,
        "size": np.int64(4),
        "alpha": np.float64(0.2),
        "beta": np.float64(0.1),
        "k": np.float64(0.9)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.local_response_norm"] = local_response_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.local_response_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.local_response_norm'.")


check_valid('torch.nn.functional.local_response_norm', generated_inputs['torch.nn.functional.local_response_norm'], lib="torch", suffix=0)
