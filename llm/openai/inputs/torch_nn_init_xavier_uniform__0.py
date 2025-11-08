
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def xavier_uniform__inputs():
    list_of_inputs = []

    tensor = torch.zeros((3, 4), dtype=torch.float32).numpy()
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.ones((128, 64), dtype=torch.float64).numpy()
    gain = np.float64(0.5)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.randn((16, 3, 3, 3), dtype=torch.float32).numpy()
    gain = np.float32(1.234)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.randn((8, 4, 5), dtype=torch.float64).numpy()
    gain = 0.9
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.randn((32, 16, 3, 3, 3), dtype=torch.float32).numpy()
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.ones((7, 7), dtype=torch.float16).numpy()
    gain = np.float16(0.7)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.full((1, 10), -1.0, dtype=torch.float32).numpy()
    gain = 2.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64).numpy()
    gain = 0.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.randn((4, 3, 2, 2, 2, 2), dtype=torch.float32).numpy()
    gain = np.float32(0.9)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64).numpy()
    gain = np.float64(1.1)
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.zeros((256, 2), dtype=torch.float16).numpy()
    gain = 0.01
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = torch.randn((3, 16, 1, 1), dtype=torch.float64).numpy()
    gain = 1.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    return list_of_inputs

generated_inputs["torch.nn.init.xavier_uniform_"] = xavier_uniform__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.xavier_uniform_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.xavier_uniform_'.")


check_valid('torch.nn.init.xavier_uniform_', generated_inputs['torch.nn.init.xavier_uniform_'], lib="torch", suffix=0)
