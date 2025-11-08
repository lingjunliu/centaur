
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def pdist_inputs():
    list_of_inputs = []

    input = torch.tensor([[1.0, -2.0, 3.5],
                          [0.0, 4.0, -1.5]], dtype=torch.float32).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                          [-1.0, -2.0, -3.0, -4.0, -5.0],
                          [0.5, -0.5, 0.5, -0.5, 0.5],
                          [10.0, 20.0, 30.0, 40.0, 50.0]], dtype=torch.float64).numpy()
    p = 1.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[-3.0],
                          [0.0],
                          [3.0]], dtype=torch.float32).numpy()
    p = 0.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, -1.0, 2.0, -2.0, 3.0],
                          [1.0, -1.0, 2.0, -2.0, 3.0],
                          [5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32).numpy()
    p = float('inf')
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64).numpy()
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[1.0, 0.0],
                          [0.5, -0.5],
                          [-1.0, 1.0],
                          [2.0, -2.0],
                          [-3.0, 3.0],
                          [4.0, -4.0],
                          [0.1, 0.2],
                          [-0.3, -0.4],
                          [5.0, 5.0],
                          [-6.0, 6.0]], dtype=torch.float32).numpy()
    p = 2.5
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[1e6, -1e6, 3e5, -3e5, 5e5, -5e5, 7e5, -7e5],
                          [-1e6, 1e6, -3e5, 3e5, -5e5, 5e5, -7e5, 7e5]], dtype=torch.float64).numpy()
    p = 3.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[1e-8, -2e-8, 3e-8],
                          [-1e-8, 2e-8, -3e-8],
                          [0.0, 0.0, 0.0],
                          [4e-8, -5e-8, 6e-8],
                          [-4e-8, 5e-8, -6e-8],
                          [1e-9, -1e-9, 2e-9]], dtype=torch.float64).numpy()
    p = 0.5
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[2.0, 2.0],
                          [2.0, -2.0],
                          [-2.0, 2.0]], dtype=torch.float64).numpy()
    p = 7.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                          [0.1, 0.2, 0.3, 0.39],
                          [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).numpy()
    p = 1e-3
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[5.0, 0.0, -5.0],
                          [5.0, 0.0, -5.0]], dtype=torch.float32).numpy()
    p = float('inf')
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    input = torch.tensor([[3.14, -2.71, 1.62, -0.57, 0.0, 10.0],
                          [-3.14, 2.71, -1.62, 0.57, 0.0, -10.0],
                          [6.28, -5.42, 3.24, -1.14, 0.0, 20.0]], dtype=torch.float64).numpy()
    p = 1.0
    list_of_inputs.append(copy.deepcopy({"input": input, "p": p}))

    return list_of_inputs

generated_inputs["torch.nn.functional.pdist"] = pdist_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pdist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pdist'.")


check_valid('torch.nn.functional.pdist', generated_inputs['torch.nn.functional.pdist'], lib="torch", suffix=0)
