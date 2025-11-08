
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def special_sinc_inputs():
    list_of_inputs = []

    input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[np.pi, np.pi / 2, -np.pi / 3],
                          [1e-6, -1e-6, 0.1]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-5, 5, steps=12, dtype=torch.float64).reshape(3, 2, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1+1j, -1-2j, 0+0j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0+1j*np.pi, 2+0j],
                          [0.5-1.5j, -3j]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([np.inf, -np.inf, np.nan, 0.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    input = base[:, ::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3, 1).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e-12, -1e-12, 1e-20, -1e-20], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base_np = torch.arange(-5, 6, dtype=torch.float32).numpy()
    input = base_np[::-2]
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.special.sinc"] = special_sinc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.sinc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.sinc'.")


check_valid('torch.special.sinc', generated_inputs['torch.special.sinc'], lib="torch", suffix=0)
