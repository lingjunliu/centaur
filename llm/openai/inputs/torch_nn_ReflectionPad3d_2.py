
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reflectionpad3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2).numpy()
    padding = (1, 1, 1, 1, 1, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 2
    input_arr = torch.randn(2, 3, 5, 3, 4, dtype=torch.float32).numpy()
    padding = (0, 2, 1, 0, 3, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 3
    input_arr = torch.arange(3*4*4*4, dtype=torch.int64).reshape(3, 4, 4, 4).numpy()
    padding = (0, 0, 0, 0, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 4
    input_arr = torch.arange(4*3*5*5, dtype=torch.int32).reshape(4, 3, 5, 5).numpy()
    padding = (2, 0, 0, 2, 1, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 5
    input_arr = torch.randn(4, 1, 7, 2, 3, dtype=torch.float32).numpy()
    padding = (1, 1, 1, 0, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 6
    input_arr = torch.randn(1, 2, 4, 5, 6, dtype=torch.float64).numpy()
    padding = (5, 0, 0, 0, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 7
    input_arr = torch.randn(2, 2, 3, 3, 3, dtype=torch.float32).numpy()
    padding = (2, 2, 2, 0, 1, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 8 (fixed reshape)
    input_arr = torch.linspace(0, 95, steps=96, dtype=torch.float32).reshape(1, 2, 3, 4, 4).numpy()
    padding = (1, 1, 2, 0, 2, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 9
    input_arr = torch.arange(1*1*2*3*4, dtype=torch.float32).reshape(1, 1, 2, 3, 4).numpy()
    padding = (0, 1, 0, 2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 10
    input_arr = torch.arange(3*5*4*4*4, dtype=torch.int64).reshape(3, 5, 4, 4, 4).numpy()
    padding = (1, 0, 3, 0, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 11
    input_arr = torch.zeros(2, 2, 2, 2, 2, dtype=torch.float32).numpy()
    padding = (0, 0, 0, 0, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 12
    input_arr = torch.randn(1, 1, 5, 6, 7, dtype=torch.float32).numpy()
    padding = (6, 0, 5, 0, 4, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad3d_2"] = reflectionpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad3d_2'.")


check_valid('torch.nn.ReflectionPad3d', generated_inputs['torch.nn.ReflectionPad3d_2'], lib="torch", suffix=2)
