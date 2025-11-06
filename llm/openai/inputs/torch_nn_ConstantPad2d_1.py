
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def constantpad2d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(3, 5, 5, dtype=torch.float32).numpy()
    padding = 0
    value = 0.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.ones(1, 3, 4, 4, dtype=torch.float32).numpy()
    padding = 1
    value = 1.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(2, 2, 3, 5, dtype=torch.float32).numpy()
    padding = np.int32(2)
    value = np.float32(-2.5)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(4, 1, 2, 2, dtype=torch.float64).numpy()
    padding = 3
    value = 5.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(1, 2, 2, dtype=torch.float32).numpy()
    padding = 4
    value = 0.1
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(1, 1, 1, 1, dtype=torch.float32).numpy()
    padding = 5
    value = 10.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(16, 3, 3, dtype=torch.float32).numpy()
    padding = np.int64(1)
    value = np.float64(3.14159)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(2, 3, 2, 3, dtype=torch.float16).numpy()
    padding = 2
    value = -0.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(2, 2, 2, dtype=torch.float32).numpy()
    padding = 6
    value = 2e-3
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(3, 4, 5, 6, dtype=torch.float32).numpy()
    padding = 1
    value = -7.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(1, 5, 1, 3, dtype=torch.float64).numpy()
    padding = 2
    value = 1000.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    input_arr = torch.randn(1, 1, 7, 7, dtype=torch.float32).numpy()
    padding = 0
    value = 42.0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "value": value, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.ConstantPad2d_1"] = constantpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConstantPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad2d_1'.")


check_valid('torch.nn.ConstantPad2d', generated_inputs['torch.nn.ConstantPad2d_1'], lib="torch", suffix=1)
