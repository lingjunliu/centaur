
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def xavier_normal__inputs():
    list_of_inputs = []

    tensor = torch.zeros((3, 4), dtype=torch.float32).numpy()
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.zeros((128, 64), dtype=torch.float64).numpy()
    gain = 0.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.zeros((32, 32), dtype=torch.float16).numpy()
    gain = 2.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = (torch.zeros((16, 16), dtype=torch.complex64)).numpy()
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = (torch.zeros((8, 8, 3, 3), dtype=torch.complex128)).numpy()
    gain = 1.41421356237
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.zeros((5, 7, 9), dtype=torch.float32).numpy()
    gain = 0.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.zeros((64, 3, 7, 7), dtype=torch.float64).numpy()
    gain = 1.2
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.zeros((2, 3, 4, 5, 6), dtype=torch.float32).numpy()
    gain = 0.9
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.zeros((6, 5), dtype=torch.float32).numpy().T
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.randn((10, 12), dtype=torch.float32).numpy()[::2, ::3]
    gain = 1.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.empty((0, 3), dtype=torch.float32).numpy()
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.empty((4, 0, 5, 5), dtype=torch.float32).numpy()
    gain = 0.7
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    return list_of_inputs

generated_inputs["torch.nn.init.xavier_normal_"] = xavier_normal__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.xavier_normal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.xavier_normal_'.")


check_valid('torch.nn.init.xavier_normal_', generated_inputs['torch.nn.init.xavier_normal_'], lib="torch", suffix=0)
