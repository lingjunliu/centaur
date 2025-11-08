
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reflectionpad2d_inputs():
    list_of_inputs = []

    padding = 0
    input_arr = torch.ones((1, 1, 1, 1), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 1
    input_arr = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 2
    input_arr = torch.randn(2, 3, 5, 6, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 1
    input_arr = torch.tensor([[[1, 2, 3, 4],
                               [5, 6, 7, 8]],
                              [[9, 10, 11, 12],
                               [13, 14, 15, 16]],
                              [[17, 18, 19, 20],
                               [21, 22, 23, 24]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 3
    input_arr = torch.randint(0, 100, (2, 4, 7, 10), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 2
    input_arr = torch.randint(0, 50, (4, 1, 3, 5), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 4
    input_arr = torch.randn(1, 1, 5, 6, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 1
    input_arr = torch.randint(-10, 10, (8, 2, 2, 2), dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 5
    input_arr = torch.linspace(0, 1, steps=2*3*6*12, dtype=torch.float32).reshape(2, 3, 6, 12).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 2
    input_arr = torch.randn(4, 5, 6, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 2
    input_arr = torch.randn(1, 5, 3, 10, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    padding = 1
    input_arr = torch.ones((10, 10, 100, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_1"] = reflectionpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad2d_1'.")


check_valid('torch.nn.ReflectionPad2d', generated_inputs['torch.nn.ReflectionPad2d_1'], lib="torch", suffix=1)
