
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def nn_init_uniform_inputs():
    list_of_inputs = []

    tensor = torch.zeros(5, dtype=torch.float32).numpy()
    a = 0.0
    b = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.ones((2, 3), dtype=torch.float32).numpy()
    a = -5.0
    b = 5.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.zeros((2, 2, 2), dtype=torch.float64).numpy()
    a = 1.5
    b = 2.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.zeros((2, 1, 3, 4), dtype=torch.float16).numpy()
    a = -0.1
    b = 0.1
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.tensor(0.0, dtype=torch.float64).numpy()
    a = 10.0
    b = 10.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.empty(0, dtype=torch.float32).numpy()
    a = -3.0
    b = -1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.zeros((4, 4, 4), dtype=torch.float32).numpy()
    a = 100.0
    b = 200.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    base = torch.arange(24, dtype=torch.float32).view(4, 6)
    tensor = base[:, ::2].numpy()
    a = -1e-3
    b = 1e-3
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.zeros((3, 7), dtype=torch.float64).numpy()
    a = -1000.0
    b = -999.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.zeros((3, 3), dtype=torch.float16).numpy()
    a = 0.001
    b = 0.002
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.zeros((1, 2, 1, 2, 3), dtype=torch.float32).numpy()
    a = 0.0
    b = 0.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    tensor = torch.zeros((2,), dtype=torch.float64).numpy()
    a = -2.5
    b = -0.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "b": b}))

    return list_of_inputs

generated_inputs["torch.nn.init.uniform_"] = nn_init_uniform_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.uniform_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.uniform_'.")


check_valid('torch.nn.init.uniform_', generated_inputs['torch.nn.init.uniform_'], lib="torch", suffix=0)
