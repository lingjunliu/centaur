
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def real_inputs():
    list_of_inputs = []

    input = torch.tensor([1+2j, -3-4j, 0+0j, -5+6j], dtype=torch.cfloat).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e-3 + 2e-3j, -7.5 + 0j, 3.14 - 2.71j], dtype=torch.cdouble).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0+1j, -2-3j, 4+0j],
                          [5-6j, -7+8j, 0-0j]], dtype=torch.cfloat).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[1+0j, -1-1j, 0+2j]],
                          [[-2+0j, 3-3j, -4+4j]]], dtype=torch.cfloat).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(3-4j, dtype=torch.cfloat).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-1.0, 0.0, 2.5, -3.3], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[-1.2, 3.4],
                          [5.6, -7.8],
                          [0.0, 9.9]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.linspace(-5, 5, steps=12, dtype=torch.float16).reshape(2, 2, 3).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([-100, 0, 255, 1024], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[0, 1, 2],
                          [253, 254, 255]], dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0,), dtype=torch.cfloat).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1+1j, 2+2j, 3+3j],
                          [4+4j, 5+5j, 6+6j]], dtype=torch.cfloat).t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([complex(float('nan'), 1.0),
                          complex(float('inf'), -float('inf')),
                          complex(-float('inf'), float('nan'))], dtype=torch.cdouble).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(42.0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.real"] = real_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.real'.")


check_valid('torch.real', generated_inputs['torch.real'], lib="torch", suffix=0)
