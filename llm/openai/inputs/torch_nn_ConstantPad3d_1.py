
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def constantpad3d_inputs():
    list_of_inputs = []
    
    input_arr = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    padding = 0
    value = 0.0
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones(1, 1, 1, 1, 1, dtype=torch.float64).numpy()
    padding = 1
    value = 1.0
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.rand(3, 2, 2, 2, dtype=torch.float32).numpy()
    padding = 2
    value = -0.5
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 3, 8, 8, 8, dtype=torch.float16).numpy()
    padding = 3
    value = 2.5
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.zeros(1, 16, 16, 16, dtype=torch.float32).numpy()
    padding = 4
    value = 10.0
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(8*1*3*7*5, dtype=torch.float32).reshape(8, 1, 3, 7, 5).numpy()
    padding = 2
    value = 0.123
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(-1.0, 1.0, steps=3*3*3*3, dtype=torch.float32).reshape(3, 3, 3, 3).numpy()
    padding = 1
    value = 100.0
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(5, 2, 9, 4, dtype=torch.float32).numpy()
    padding = 5
    value = -3.14
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(10, 4, 5, 6, dtype=torch.float64).numpy()
    padding = 2
    value = 0.001
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 64, 32, 32, 32, dtype=torch.float32).numpy()
    padding = 6
    value = 5.0
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 8, 3, 3, 3, dtype=torch.float32).numpy()
    padding = 1
    value = -0.0
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(7, 7, 7, 7, dtype=torch.float32).numpy()
    padding = 3
    value = 42.42
    input_dict = {"padding": padding, "value": value, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ConstantPad3d_1"] = constantpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConstantPad3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad3d_1'.")


check_valid('torch.nn.ConstantPad3d', generated_inputs['torch.nn.ConstantPad3d_1'], lib="torch", suffix=1)
