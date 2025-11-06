
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_avg_pool3d_inputs():
    list_of_inputs = []

    x = torch.arange(4*4*4, dtype=torch.float32).reshape(1, 1, 4, 4, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 1}))

    x = torch.randn(2, 3, 5, 7, 9, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 2}))

    x = torch.zeros(3, 1, 6, 5, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 2}))

    x = torch.linspace(-1, 1, steps=1*4*1*7*5, dtype=torch.float32).reshape(1, 4, 1, 7, 5).numpy()
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 1}))

    x = np.full((5, 6, 2, 2, 2), -3.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 2}))

    t = torch.rand(2, 2, 9, 1, 3, dtype=torch.float32)
    t = (t - 0.5) * 2.0
    x = t.numpy()
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 1}))

    x = torch.randn(1, 3, 8, 8, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 8}))

    x = np.random.standard_normal((2, 1, 16, 10, 6)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 6}))

    x = torch.randn(1, 5, 3, 8, 2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 2}))

    x = (np.arange(7*2*4*3*5, dtype=np.float32).reshape(7, 2, 4, 3, 5) * 0.1) - 5.0
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 3}))

    x = np.random.randn(3, 10, 12, 14).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 4}))

    x = np.random.randn(1, 2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"input": x, "output_size": 1}))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_avg_pool3d_2"] = adaptive_avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_avg_pool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_avg_pool3d_2'.")


check_valid('torch.nn.functional.adaptive_avg_pool3d', generated_inputs['torch.nn.functional.adaptive_avg_pool3d_2'], lib="torch", suffix=2)
