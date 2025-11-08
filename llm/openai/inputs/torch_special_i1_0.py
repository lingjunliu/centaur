
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_special_i1_inputs():
    list_of_inputs = []

    input = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[-10.0, -5.0, 0.0], [5.0, 10.0, 20.0]], dtype=torch.float64).numpy()
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.linspace(-5.0, 5.0, steps=24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor(2.5, dtype=torch.float32).numpy()
    out = np.empty((), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.randn(2, 1, 3, 4, dtype=torch.float64).numpy()
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.empty(0, dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.empty(2, 0, dtype=torch.float64).numpy()
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.arange(24, dtype=torch.float32).reshape(4, 6).numpy()
    input = base[:, ::2]
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([np.nan, np.inf, -np.inf, 0.0, 1.0], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([1e-12, -1e-12, 1e-6, -1e-6], dtype=torch.float64).numpy()
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([30.0, 50.0, 80.0], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    a = torch.linspace(-3.0, 3.0, steps=12, dtype=torch.float32).reshape(3, 4).numpy()
    input = np.asfortranarray(a)
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.i1"] = torch_special_i1_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.i1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.i1'.")


check_valid('torch.special.i1', generated_inputs['torch.special.i1'], lib="torch", suffix=0)
