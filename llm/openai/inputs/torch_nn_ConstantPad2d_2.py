
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def constantpad2d_inputs():
    list_of_inputs = []

    # 1
    input = torch.randn(1, 2, 2).numpy()
    padding = (1, 1, 1, 1)
    value = 0.0
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 2
    input = torch.randn(3, 4, 5).numpy()
    padding = (2, 0, 1, 3)
    value = 1.0
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 3
    input = torch.randn(2, 3, 5, 4).numpy()
    padding = (0, 2, 0, 2)
    value = -2.5
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 4
    input = torch.randn(1, 1, 1, 1).numpy()
    padding = (3, 3, 3, 3)
    value = 10.0
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 5
    input = torch.randn(2, 1, 3).numpy()
    padding = (0, 1, 2, 0)
    value = 3.14
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 6
    input = torch.randn(4, 2, 7, 7, dtype=torch.float16).numpy()
    padding = (1, 2, 3, 4)
    value = 0.5
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 7
    input = torch.randn(5, 3, 3, dtype=torch.float64).numpy()
    padding = (0, 0, 0, 0)
    value = -0.0
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 8
    input = torch.randn(1, 4, 8, 2).numpy()
    padding = (5, 0, 0, 5)
    value = 7.777
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 9
    input = torch.randn(1, 10, 1).numpy()
    padding = (1, 0, 4, 4)
    value = -100.0
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 10
    input = torch.randn(3, 1, 2, 6).numpy()
    padding = (2, 2, 0, 1)
    value = 2.71828
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 11
    input = torch.randn(8, 8, 8).numpy()
    padding = (4, 4, 4, 4)
    value = 1e-6
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    # 12
    input = torch.randn(2, 5, 3, 3).numpy()
    padding = (0, 1, 5, 0)
    value = -0.25
    list_of_inputs.append(copy.deepcopy({"input": input, "padding": padding, "value": value}))

    return list_of_inputs

generated_inputs["torch.nn.ConstantPad2d_2"] = constantpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConstantPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad2d_2'.")


check_valid('torch.nn.ConstantPad2d', generated_inputs['torch.nn.ConstantPad2d_2'], lib="torch", suffix=2)
