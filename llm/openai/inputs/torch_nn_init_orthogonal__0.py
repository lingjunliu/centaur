
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def orthogonal_inputs():
    list_of_inputs = []

    tensor = torch.zeros((3, 3), dtype=torch.float32).numpy()
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = np.random.randn(5, 3).astype(np.float64)
    gain = 0.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = np.random.randn(3, 5).astype(np.float64)
    gain = 2.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = (np.random.randn(2, 2).astype(np.float32) * -3.0)
    gain = -1.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = np.ones((4, 4), dtype=np.float32)
    gain = 0.0
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = np.random.randn(2, 3, 3).astype(np.float32)
    gain = float(np.sqrt(2.0))
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = np.random.randn(2, 3, 5).astype(np.float64)
    gain = 3.141592653589793
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = np.random.randn(6, 4).astype(np.float64)
    gain = 1e-3
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = (np.random.randn(6, 1).astype(np.float32) - 5.0)
    gain = 1.5
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = (np.random.randn(1, 6).astype(np.float32) * 2.0)
    gain = -0.7
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = (np.arange(100).reshape(10, 10).astype(np.float64) - 50.0)
    gain = 1.2
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    tensor = np.zeros((8, 2, 2), dtype=np.float32)
    gain = 0.8
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "gain": gain}))

    return list_of_inputs

generated_inputs["torch.nn.init.orthogonal_"] = orthogonal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.orthogonal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.orthogonal_'.")


check_valid('torch.nn.init.orthogonal_', generated_inputs['torch.nn.init.orthogonal_'], lib="torch", suffix=0)
