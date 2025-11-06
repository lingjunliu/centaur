
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def ravel_inputs():
    list_of_inputs = []

    input = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[[-1.5, 2.0], [3.3, -4.4]], [[5.5, -6.6], [7.7, 8.8]]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(42).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1+2j, -3+0.5j, 0-1j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((0, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(12, dtype=torch.int64).reshape(3, 4).t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4).permute(2, 0, 1).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    input = base[::2, :].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[float('nan'), float('inf')], [-float('inf'), 0.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randint(-10, 10, (2, 1, 3, 1, 4), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(3, 3, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty((2, 0, 3, 4), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1+0j, 0-2j, -3+4j], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.ravel"] = ravel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ravel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ravel'.")


check_valid('torch.ravel', generated_inputs['torch.ravel'], lib="torch", suffix=0)
