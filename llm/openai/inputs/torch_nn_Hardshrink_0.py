
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def hardshrink_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.tensor([-1.0, -0.4, 0.0, 0.2, 1.2], dtype=torch.float32).numpy()
    input_dict = {"lambd": 0.5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.tensor([[0.05, -0.05, 0.2], [-0.2, 1.0, -1.0]], dtype=torch.float32).numpy()
    input_dict = {"lambd": 0.1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3 (0-D scalar)
    input_arr = torch.tensor(0.3, dtype=torch.float32).numpy()
    input_dict = {"lambd": 0.5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4 (3D, float64)
    input_arr = (torch.randn((2, 2, 3), dtype=torch.float64) * 2.0).numpy()
    input_dict = {"lambd": 1.0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5 (4D, lambd=0.0)
    input_arr = torch.randn((1, 3, 4, 2), dtype=torch.float32).numpy()
    input_dict = {"lambd": 0.0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6 (threshold checks with equals)
    input_arr = torch.tensor([-5.0, -2.0, -2.5, 2.4, 2.5, 5.0], dtype=torch.float32).numpy()
    input_dict = {"lambd": 2.5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7 (2D grid)
    input_arr = torch.linspace(-1.5, 1.5, steps=9, dtype=torch.float32).reshape(3, 3).numpy()
    input_dict = {"lambd": 0.75, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8 (small magnitudes)
    input_arr = torch.tensor([0.1, -0.2, 0.9, -0.99], dtype=torch.float32).numpy()
    input_dict = {"lambd": 1.0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9 (float16)
    input_arr = torch.tensor([[0.1, -0.3, 0.25, -0.05], [1.2, -1.3, 0.0, 0.5]], dtype=torch.float16).numpy()
    input_dict = {"lambd": 0.2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10 (empty dimension)
    input_arr = torch.empty((3, 0), dtype=torch.float32).numpy()
    input_dict = {"lambd": 0.3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11 (all zeros)
    input_arr = torch.zeros((5,), dtype=torch.float32).numpy()
    input_dict = {"lambd": 0.5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12 (boundary values around lambd)
    lambd_val = 1.5
    input_arr = torch.tensor([-lambd_val, lambd_val, -lambd_val - 1e-4, lambd_val + 1e-4, 0.0], dtype=torch.float32).numpy()
    input_dict = {"lambd": lambd_val, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 13 (very small lambda)
    input_arr = torch.tensor([0.0, 1e-7, -1e-7, 1e-5, -1e-5], dtype=torch.float32).numpy()
    input_dict = {"lambd": 1e-6, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 14 (3D float32, mixed)
    input_arr = torch.tensor([[[-0.4, 0.4], [2.0, -2.0]], [[0.6, -0.6], [0.49, -0.49]]], dtype=torch.float32).numpy()
    input_dict = {"lambd": 0.5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.Hardshrink"] = hardshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Hardshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Hardshrink'.")


check_valid('torch.nn.Hardshrink', generated_inputs['torch.nn.Hardshrink'], lib="torch", suffix=0)
