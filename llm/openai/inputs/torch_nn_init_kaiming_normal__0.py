
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def kaiming_normal_inputs():
    list_of_inputs = []

    tensor = torch.randn((3, 5), dtype=torch.float32).numpy()
    a = 0.0
    mode = "fan_in"
    nonlinearity = "relu"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((64, 3, 7, 7), dtype=torch.float32).numpy()
    a = 0.2
    mode = "fan_out"
    nonlinearity = "leaky_relu"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((10, 10), dtype=torch.float64).numpy()
    a = -0.1
    mode = "fan_in"
    nonlinearity = "tanh"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((16, 32, 8), dtype=torch.float32).numpy()
    a = 0.5
    mode = "fan_out"
    nonlinearity = "selu"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((32, 16, 3, 3, 3), dtype=torch.float32).numpy()
    a = 1.0
    mode = "fan_in"
    nonlinearity = "leaky_relu"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((3, 64, 5, 5), dtype=torch.float32).numpy()
    a = 0.01
    mode = "fan_out"
    nonlinearity = "linear"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((128, 256), dtype=torch.float32).numpy()
    a = 0.0
    mode = "fan_in"
    nonlinearity = "relu"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((4, 4), dtype=torch.float64).numpy()
    a = 2.0
    mode = "fan_out"
    nonlinearity = "sigmoid"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((32, 16, 5), dtype=torch.float32).numpy()
    a = 0.3
    mode = "fan_in"
    nonlinearity = "relu"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((2, 3, 4, 5, 6, 7), dtype=torch.float32).numpy()
    a = 0.2
    mode = "fan_out"
    nonlinearity = "leaky_relu"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((8, 8), dtype=torch.float16).numpy()
    a = 0.0
    mode = "fan_out"
    nonlinearity = "relu"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    tensor = torch.randn((6, 12, 1, 1), dtype=torch.float32).numpy()
    a = 0.25
    mode = "fan_in"
    nonlinearity = "leaky_relu"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "a": a, "mode": mode, "nonlinearity": nonlinearity}))

    return list_of_inputs

generated_inputs["torch.nn.init.kaiming_normal_"] = kaiming_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.kaiming_normal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.kaiming_normal_'.")


check_valid('torch.nn.init.kaiming_normal_', generated_inputs['torch.nn.init.kaiming_normal_'], lib="torch", suffix=0)
