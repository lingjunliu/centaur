
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def reflectionpad3d_inputs():
    list_of_inputs = []

    # Input 1
    inp = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2).numpy()
    padding = 1
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 2
    inp = torch.randn(1, 1, 3, 4, 5, dtype=torch.float64).numpy()
    padding = 2
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 3
    inp = torch.arange(3*5*6*7, dtype=torch.int32).reshape(3, 5, 6, 7).numpy()
    padding = 1
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 4
    inp = torch.arange(2*3*4*4*4, dtype=torch.int16).reshape(2, 3, 4, 4, 4).numpy()
    padding = 1
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 5
    inp = torch.ones((1, 1, 1, 1, 1), dtype=torch.float32).numpy()
    padding = 0
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 6
    inp = torch.randn(2, 5, 8, 6, dtype=torch.float64).numpy()
    padding = 3
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 7
    inp = torch.randint(0, 256, (4, 1, 2, 3, 2)).to(torch.uint8).numpy()
    padding = 1
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 8
    inp = torch.rand(1, 4, 4, 6, dtype=torch.float32).to(torch.float16).numpy()
    padding = 2
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 9
    inp = torch.randn(1, 2, 6, 7, 5, dtype=torch.float32).numpy()
    padding = 4
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 10
    inp = torch.linspace(-10, 10, steps=1*2*2*3, dtype=torch.float32).reshape(1, 2, 2, 3).numpy()
    padding = 1
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 11
    inp = torch.arange(3*2*3*5*4, dtype=torch.int64).reshape(3, 2, 3, 5, 4).numpy()
    padding = 2
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    # Input 12
    inp = torch.arange(1*1*6*7*10, dtype=torch.float32).reshape(1, 1, 6, 7, 10).numpy()
    padding = 5
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad3d_1"] = reflectionpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad3d_1'.")


check_valid('torch.nn.ReflectionPad3d', generated_inputs['torch.nn.ReflectionPad3d_1'], lib="torch", suffix=1)
