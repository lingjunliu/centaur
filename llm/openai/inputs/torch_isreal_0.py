
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def isreal_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, -2.5, float('nan'), float('inf')], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1+0j, 1+2j, -3-0j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[-1, 0, 2], [3, -4, 5]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(3+0j, dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    real = torch.randn(2, 3, 4, dtype=torch.float32)
    imag = torch.randn(2, 3, 4, dtype=torch.float32)
    input = (real + 1j * imag).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty(0, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(4, 5, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    base = torch.arange(24, dtype=torch.float32).view(4, 6)
    input = base[:, ::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.zeros(2, 0, 3, 1, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([0+0j, -1+0j, 2.5+0j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1e20 + 1e-30j, -1e-30 + 2j, 3 - 0j], dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(12, dtype=torch.uint8).reshape(3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randint(-128, 128, (3, 3, 3), dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.isreal"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isreal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isreal'.")


check_valid('torch.isreal', generated_inputs['torch.isreal'], lib="torch", suffix=0)
