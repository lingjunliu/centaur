
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def mish_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": input}))

    input = torch.randn(2, 3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": input}))

    input = (torch.randn(1, 2, 3, dtype=torch.float16) * 5).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": input}))

    input = torch.tensor(3.14159, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": input}))

    input = torch.tensor([100.0, -100.0, 1e-6, -1e-6], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": input}))

    input = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": input}))

    input = torch.randn(2, 1, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": input}))

    input = (torch.randn(1, 2, 1, 1, 2, dtype=torch.float32) - 2.0).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": input}))

    input = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": input}))

    base = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
    input = base[:, ::2, :].numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": input}))

    input = (-torch.ones((3, 3), dtype=torch.float64)).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": input}))

    input = torch.randn(64, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": input}))

    input = torch.empty(2, 0, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": input}))

    input = torch.rand(4, 5, 6, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": input}))

    return list_of_inputs

generated_inputs["torch.nn.Mish"] = mish_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Mish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Mish'.")


check_valid('torch.nn.Mish', generated_inputs['torch.nn.Mish'], lib="torch", suffix=0)
