
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def softshrink_inputs():
    list_of_inputs = []

    input = torch.tensor([-1.5, -0.5, 0.0, 0.3, 0.5, 1.0, 2.5], dtype=torch.float32).numpy()
    lambd = 0.5
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-2.0, -1.0, -0.9],
                          [0.9, 1.0, 2.0]], dtype=torch.float64).numpy()
    lambd = 1.0
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor(0.3, dtype=torch.float32).numpy()
    lambd = 0.5
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[-1.0, -0.1],
                           [0.0, 0.1]],
                          [[0.9, -0.9],
                           [1.1, -1.1]]], dtype=torch.float32).numpy()
    lambd = 0.0
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-5, 5, steps=11, dtype=torch.float32).numpy()
    lambd = 3.0
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([-0.3, 0.2, 0.4, 1.5], dtype=torch.float16).numpy()
    lambd = 0.2
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(-12, 12, dtype=torch.float32).reshape(2, 1, 3, 4).numpy()
    lambd = 0.75
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    lambd = 0.5
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-0.3],
                          [0.0],
                          [0.7]], dtype=torch.float64).numpy()
    lambd = 0.25
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2, 1).numpy()
    lambd = 0.5
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-5.0, -2.5, -0.2],
                          [-0.7, -1.5, -3.3]], dtype=torch.float32).numpy()
    lambd = 2.0
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([1e-6, -1e-6, 1e-4], dtype=torch.float32).numpy()
    lambd = 1e-5
    input_dict = {"input": input, "lambd": lambd}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.softshrink"] = softshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.softshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.softshrink'.")


check_valid('torch.nn.functional.softshrink', generated_inputs['torch.nn.functional.softshrink'], lib="torch", suffix=0)
