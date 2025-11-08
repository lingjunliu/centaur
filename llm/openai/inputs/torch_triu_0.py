
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def triu_inputs():
    list_of_inputs = []

    t = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    input_dict = {"input": t.numpy(), "diagonal": 0, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.linspace(-5, 5, steps=24, dtype=torch.float64).reshape(4, 6)
    input_dict = {"input": t.numpy(), "diagonal": 1, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randint(-5, 6, (6, 4), dtype=torch.int32)
    input_dict = {"input": t.numpy(), "diagonal": -1, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn(2, 3, 3, dtype=torch.float32)
    input_dict = {"input": t.numpy(), "diagonal": 0, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randint(-100, 100, (2, 2, 4, 4), dtype=torch.int64)
    input_dict = {"input": t.numpy(), "diagonal": 2, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randint(0, 2, (5, 5), dtype=torch.bool)
    input_dict = {"input": t.numpy(), "diagonal": 0, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = (torch.randn(3, 5) + 1j * torch.randn(3, 5)).to(torch.complex64)
    input_dict = {"input": t.numpy(), "diagonal": -2, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn(0, 0, dtype=torch.float32)
    input_dict = {"input": t.numpy(), "diagonal": 0, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn(0, 7, dtype=torch.float64)
    input_dict = {"input": t.numpy(), "diagonal": -3, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn(7, 0, dtype=torch.float32)
    input_dict = {"input": t.numpy(), "diagonal": 0, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.tensor([[42]], dtype=torch.int32)
    input_dict = {"input": t.numpy(), "diagonal": 5, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = (torch.randn(2, 3, 4, dtype=torch.float64) + 1j * torch.randn(2, 3, 4, dtype=torch.float64)).to(torch.complex128)
    input_dict = {"input": t.numpy(), "diagonal": 10, "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.triu"] = triu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.triu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.triu'.")


check_valid('torch.triu', generated_inputs['torch.triu'], lib="torch", suffix=0)
