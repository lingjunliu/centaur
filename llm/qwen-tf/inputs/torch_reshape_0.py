
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reshape_inputs():
    list_of_inputs = []

    input = torch.arange(6, dtype=torch.float32).numpy()
    shape = (2, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64).numpy()
    shape = (-1,)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = (torch.arange(24) % 2 == 0).view(2, 3, 4).numpy()
    shape = (3, 8)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([1+2j, 3+4j, 5+6j, 7+8j], dtype=torch.complex64).view(1, 1, 4).numpy()
    shape = (2, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([], dtype=torch.float32).numpy()
    shape = (0, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.randn(2, 1, 3, 4, dtype=torch.float64).numpy()
    shape = (-1, 2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    t = torch.arange(12, dtype=torch.int32).view(6, 2)
    input = t[:, 0].numpy()
    shape = (2, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=torch.int16).numpy()
    shape = (5, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.arange(32, dtype=torch.float16).view(2, 4, 4).numpy()
    shape = (4, 2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor(5.0, dtype=torch.float32).numpy()
    shape = (1,)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.arange(24, dtype=torch.int32).numpy()
    shape = (2, -1, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    input = torch.tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], dtype=torch.complex128).numpy()
    shape = (-1,)
    list_of_inputs.append(copy.deepcopy({"input": input, "shape": shape}))

    return list_of_inputs

generated_inputs["torch.reshape"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.reshape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.reshape'.")


check_valid('torch.reshape', generated_inputs['torch.reshape'], lib="torch", suffix=0)
