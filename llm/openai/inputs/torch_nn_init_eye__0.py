
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def nn_init_eye__inputs():
    list_of_inputs = []

    tensor = torch.zeros((2, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = torch.randn((3, 5), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = torch.ones((5, 3), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = (torch.randn((1, 1), dtype=torch.float32) + 1j * torch.randn((1, 1), dtype=torch.float32)).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    real = torch.randn((4, 4), dtype=torch.float64)
    imag = torch.randn((4, 4), dtype=torch.float64)
    tensor = (real + 1j * imag).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = torch.empty((0, 0), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = torch.arange(100, dtype=torch.float32).reshape(10, 10).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    base = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    tensor = base[::2, ::2].numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = torch.randn((1, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = torch.randn((5, 1), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = torch.linspace(-5, 5, steps=64*32, dtype=torch.float32).reshape(64, 32).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    tensor = torch.linspace(0, 1, steps=32*64, dtype=torch.float64).reshape(32, 64).numpy()
    list_of_inputs.append(copy.deepcopy({"tensor": tensor}))

    return list_of_inputs

generated_inputs["torch.nn.init.eye_"] = nn_init_eye__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.eye_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.eye_'.")


check_valid('torch.nn.init.eye_', generated_inputs['torch.nn.init.eye_'], lib="torch", suffix=0)
