
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def isnan_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, float('nan'), -3.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.0, 2.0, float('nan')], [float('nan'), -5.0, 6.5]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(float('nan'), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[0.0, -1.0], [float('nan'), 3.0]], [[4.0, 5.0], [6.0, float('nan')]]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([complex(1.0, 2.0), complex(float('nan'), 1.0), complex(2.0, float('nan')), complex(float('nan'), float('nan'))], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[complex(0.0, 0.0), complex(1.0, -1.0)], [complex(float('nan'), 0.0), complex(2.5, float('nan'))]], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('inf'), -float('inf'), 0.0, 1.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.linspace(-3.0, 3.0, steps=7, dtype=torch.float32)
    base[3] = float('nan')
    input = base[::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((2, 0, 3, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-1e308, 1e308, float('nan'), -0.0, 0.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0,), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[float('nan')]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    t = torch.zeros((1, 2, 1, 3, 2), dtype=torch.float32)
    t[0, 1, 0, 2, 1] = float('nan')
    input = t.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.isnan"] = isnan_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isnan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isnan'.")


check_valid('torch.isnan', generated_inputs['torch.isnan'], lib="torch", suffix=0)
