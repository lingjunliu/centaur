
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reflectionpad1d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).numpy()
    padding = (2, 2)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 2
    input_arr = torch.tensor([
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        [2.0, -2.0, 3.5, -3.5, 0.0],
        [7.1, 8.2, -9.3, 10.4, -11.5]
    ], dtype=torch.float64).numpy()
    padding = (1, 3)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 3
    input_arr = torch.linspace(0, 1, steps=40, dtype=torch.float32).reshape(4, 1, 10).numpy()
    padding = (0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 4
    input_arr = torch.randn(2, 2, 3, dtype=torch.float32).numpy()
    padding = (1, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 5
    input_arr = torch.linspace(-3, 3, steps=35, dtype=torch.float32).reshape(5, 7).numpy()
    padding = (2, 4)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 6
    input_arr = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2).numpy()
    padding = (1, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 7
    input_arr = torch.sin(torch.linspace(-3.14, 3.14, steps=16, dtype=torch.float64)).reshape(2, 8).numpy()
    padding = (7, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 8
    input_arr = torch.rand(3, 3, 6, dtype=torch.float32).numpy()
    padding = (0, 5)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 9
    input_arr = torch.linspace(0, 10, steps=36, dtype=torch.float64).reshape(4, 9).numpy()
    padding = (0, 8)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 10
    input_arr = torch.linspace(0.0, 1.0, steps=50, dtype=torch.float32).reshape(1, 1, 50).numpy()
    padding = (25, 24)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 11
    input_arr = torch.randn(5, 2, 4, dtype=torch.float32).numpy()
    padding = (3, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 12
    input_arr = torch.arange(36, dtype=torch.float64).reshape(6, 6).numpy()
    padding = (5, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad1d_2"] = reflectionpad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad1d_2'.")


check_valid('torch.nn.ReflectionPad1d', generated_inputs['torch.nn.ReflectionPad1d_2'], lib="torch", suffix=2)
