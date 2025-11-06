
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def arctanh_inputs():
    list_of_inputs = []

    input = torch.tensor([-0.8, -0.2, 0.0, 0.2, 0.8], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[-0.999999, 0.999999], [0.1, -0.1]], dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (torch.rand(2, 3, 4, dtype=torch.float16) * 1.8 - 0.9).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor(0.5, dtype=torch.float64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = torch.linspace(-0.9, 0.9, 6, dtype=torch.float32).reshape(2, 3)
    input = base.transpose(0, 1).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([0.5 + 0.2j, -0.3 + 0.4j, 0.0 + 0.0j], dtype=torch.complex64).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([[0.1 + 0.9j, -0.9 + 0.1j], [0.5 - 0.5j, -0.1 - 0.2j]], dtype=torch.complex128).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = torch.tensor([float('nan'), 0.0, float('inf'), -float('inf'), 1.0, -1.0, 0.5, -0.5], dtype=torch.float32).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base_np = np.linspace(-0.9, 0.9, 24, dtype=np.float64).reshape(4, 6)
    input = base_np[:, ::2]
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    input = (torch.rand(2, 1, 3, 3, dtype=torch.float32) * 1.98 - 0.99).numpy()
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    rs = np.random.RandomState(0)
    input = rs.uniform(-0.95, 0.95, size=20).astype(np.float64)
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.arctanh"] = arctanh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.arctanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arctanh'.")


check_valid('torch.arctanh', generated_inputs['torch.arctanh'], lib="torch", suffix=0)
