
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def glu_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(4, 8, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.arange(20, dtype=torch.float32).reshape(10, 2).numpy()
    input_dict = {"input": input_arr, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(3, 6, 5, dtype=torch.float32).to(torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.linspace(-5, 5, steps=48, dtype=torch.float32).reshape(2, 3, 8).numpy()
    input_dict = {"input": input_arr, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.randn(2, 4, 6, 8, dtype=torch.float32).add_(-0.5).numpy()
    input_dict = {"input": input_arr, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.arange(40, dtype=torch.float64).reshape(1, 2, 2, 10).numpy()
    input_dict = {"input": input_arr, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.rand(2, 2, 2, 2, 4, dtype=torch.float32).mul_(2).sub_(1).numpy()
    input_dict = {"input": input_arr, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.randn(3, 5, 7, 9, 12, dtype=torch.float32).to(torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.sin(torch.linspace(-3.14, 3.14, steps=100, dtype=torch.float64)).numpy()
    input_dict = {"input": input_arr, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.linspace(-1, 1, steps=84, dtype=torch.float32).reshape(6, 14).to(torch.float16).numpy()
    input_dict = {"input": input_arr, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn(7, 14, 3, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "dim": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.glu"] = glu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.glu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.glu'.")


check_valid('torch.nn.functional.glu', generated_inputs['torch.nn.functional.glu'], lib="torch", suffix=0)
