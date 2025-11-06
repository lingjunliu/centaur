
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def constantpad1d_inputs():
    list_of_inputs = []

    input = torch.randn(2, 4, dtype=torch.float32).numpy()
    padding = 2
    value = 3.5
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(3, 5, dtype=torch.float64).numpy()
    padding = 0
    value = 0.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(1, 3, 4, dtype=torch.float32).numpy()
    padding = 1
    value = -1.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(5, 10, dtype=torch.float16).numpy()
    padding = 4
    value = 2.25
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(2, 1, 1, dtype=torch.float32).numpy()
    padding = 3
    value = 100.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(4, 3, 7, dtype=torch.float64).numpy()
    padding = 5
    value = -0.5
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(1, 1, dtype=torch.float32).numpy()
    padding = 1
    value = 42.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(2, 3, 1, dtype=torch.float64).numpy()
    padding = 8
    value = 1e-3
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(7, 9, dtype=torch.float32).numpy()
    padding = 2
    value = 0.75
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(3, 2, 5, dtype=torch.float32).numpy()
    padding = 6
    value = -3.14159
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(64, 128, dtype=torch.float32).numpy()
    padding = 10
    value = 1.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    input = torch.randn(1, 4, 2, dtype=torch.float16).numpy()
    padding = 9
    value = 5e2
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input}))

    return list_of_inputs

generated_inputs["torch.nn.ConstantPad1d_1"] = constantpad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConstantPad1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad1d_1'.")


check_valid('torch.nn.ConstantPad1d', generated_inputs['torch.nn.ConstantPad1d_1'], lib="torch", suffix=1)
