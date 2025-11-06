
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def hsplit_inputs():
    list_of_inputs = []

    input = torch.linspace(-3.0, 4.0, steps=8, dtype=torch.float32).numpy()
    indices_or_sections = (2, 5)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.empty((0,), dtype=torch.int32).numpy()
    indices_or_sections = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(18., dtype=torch.float64).reshape(3, 6).numpy()
    indices_or_sections = (2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(-10, 0, dtype=torch.int64).reshape(2, 5).numpy()
    indices_or_sections = (3,)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(16, dtype=torch.float32).reshape(4, 4).numpy()
    indices_or_sections = (0, 2, 2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = (torch.arange(24).reshape(2, 4, 3) % 2 == 0).numpy()
    indices_or_sections = (2, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(56, dtype=torch.float16).reshape(2, 7, 4).numpy()
    indices_or_sections = (1, 6)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(30, dtype=torch.uint8).reshape(3, 5, 2).numpy()
    indices_or_sections = (1, 4, 5)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.randn((1, 5, 2, 2), dtype=torch.float32).numpy()
    indices_or_sections = (1, 4)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    real = torch.randn((2, 8, 1, 1), dtype=torch.float32)
    imag = torch.randn((2, 8, 1, 1), dtype=torch.float32)
    input = torch.complex(real, imag).numpy()
    indices_or_sections = (3, 8)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.int16).reshape(2, 3, 4, 5, 6).numpy()
    indices_or_sections = (1, 3)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.empty((4, 0), dtype=torch.float32).numpy()
    indices_or_sections = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    return list_of_inputs

generated_inputs["torch.hsplit_3"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.hsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hsplit_3'.")


check_valid('torch.hsplit', generated_inputs['torch.hsplit_3'], lib="torch", suffix=3)
