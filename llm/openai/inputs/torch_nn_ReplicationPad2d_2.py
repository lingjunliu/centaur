
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def replicationpad2d_inputs():
    list_of_inputs = []

    # 1
    input = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).numpy()
    padding = (2, 2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 2
    input = torch.randn(2, 3, 5, 4, dtype=torch.float64).numpy()
    padding = (0, 0, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 3
    input = torch.randint(-10, 10, (3, 2, 2), dtype=torch.int64).numpy()
    padding = (1, 1, 1, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 4
    input = torch.randint(-5, 5, (1, 2, 1, 4), dtype=torch.int64).to(torch.int32).numpy()
    padding = (0, 2, 3, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 5
    input = torch.randn(1, 1, 1, dtype=torch.float16).numpy()
    padding = (1, 0, 0, 2)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 6
    input = torch.randint(0, 100, (4, 1, 10, 1), dtype=torch.int64).to(torch.int16).numpy()
    padding = (3, 1, 0, 5)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 7
    input = torch.randint(-50, 50, (2, 4, 3), dtype=torch.int64).to(torch.int32).numpy()
    padding = (2, 0, 1, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 8
    input = torch.randn(3, 2, 7, 5, dtype=torch.float32).numpy()
    padding = (0, 1, 2, 2)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 9
    input = torch.linspace(-1.0, 1.0, steps=30, dtype=torch.float32).reshape(2, 3, 5).numpy()
    padding = (5, 0, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 10
    input = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2).numpy()
    padding = (1, 2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 11
    input = torch.randint(-100, 100, (1, 1, 2, 3), dtype=torch.int64).numpy()
    padding = (4, 0, 4, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    # 12
    input = torch.randn(1, 1, 2, 2, dtype=torch.float32).numpy()
    padding = (10, 0, 0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input}))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad2d_2"] = replicationpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad2d_2'.")


check_valid('torch.nn.ReplicationPad2d', generated_inputs['torch.nn.ReplicationPad2d_2'], lib="torch", suffix=2)
