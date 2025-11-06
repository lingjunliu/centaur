
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def constantpad1d_inputs():
    list_of_inputs = []

    input = torch.randn(3, 5, dtype=torch.float32).numpy()
    padding = (2, 2)
    value = 0.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(1, 2, 4, dtype=torch.float32).numpy()
    padding = (1, 3)
    value = 3.5
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(1, 0, dtype=torch.float32).numpy()
    padding = (2, 2)
    value = 1.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(4, 3, 7, dtype=torch.float32).numpy()
    padding = (0, 0)
    value = 2.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(2, 1, 1, dtype=torch.float64).numpy()
    padding = (5, 1)
    value = -1.25
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(2, 8, dtype=torch.float16).numpy()
    padding = (3, 0)
    value = 0.5
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(3, 5, 2, dtype=torch.float32).numpy()
    padding = (1, 1)
    value = 10.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(4, 3, dtype=torch.float32).numpy()
    padding = (0, 4)
    value = -100.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(1, 1, 10, dtype=torch.float32).numpy()
    padding = (9, 0)
    value = 0.123456789
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(2, 1, dtype=torch.float32).numpy()
    padding = (1, 1)
    value = 7.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(2, 2, 3, dtype=torch.float32).numpy()
    padding = (2, 2)
    value = -0.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    input = torch.randn(1, 4, 6, dtype=torch.float32).numpy()
    padding = (6, 6)
    value = 42.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": float(value), "input": input}))

    return list_of_inputs

generated_inputs["torch.nn.ConstantPad1d_2"] = constantpad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConstantPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad1d_2'.")


check_valid('torch.nn.ConstantPad1d', generated_inputs['torch.nn.ConstantPad1d_2'], lib="torch", suffix=2)
