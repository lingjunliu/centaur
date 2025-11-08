
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def dsplit_inputs():
    list_of_inputs = []

    input = torch.arange(16.0, dtype=torch.float32).reshape(2, 2, 4).numpy()
    indices_or_sections = [2]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(-27, 0, dtype=torch.int32).reshape(3, 3, 3).numpy()
    indices_or_sections = [1, 2]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(20, dtype=torch.int16).reshape(1, 4, 5).numpy()
    indices_or_sections = [-2]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(6, dtype=torch.int64).reshape(2, 1, 3).numpy()
    indices_or_sections = [0, 0, 2]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.randn(2, 3, 6, 4, dtype=torch.float64).numpy()
    indices_or_sections = [2, 4]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = (torch.rand(1, 2, 1, 3, 4) > 0.5).numpy()
    indices_or_sections = [1, 1, 1]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.randn(2, 2, 2, dtype=torch.complex64).numpy()
    indices_or_sections = [1]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.empty(3, 2, 0, dtype=torch.int64).numpy()
    indices_or_sections = [0, 0]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    base = torch.arange(3 * 3 * 8 * 2, dtype=torch.float32).reshape(3, 3, 8, 2)
    input = base[:, :, ::2, :].numpy()
    indices_or_sections = [2, 3, 10]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).permute(1, 0, 2).numpy()
    indices_or_sections = [-4, -1]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.randn(1, 1, 7, 1, 1, 1, dtype=torch.float16).numpy()
    indices_or_sections = [3, 6]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    input = torch.ones(2, 2, 3, dtype=torch.uint8).numpy()
    indices_or_sections = [10]
    list_of_inputs.append(copy.deepcopy({"input": input, "indices_or_sections": indices_or_sections}))

    return list_of_inputs

generated_inputs["torch.dsplit_2"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dsplit_2'.")


check_valid('torch.dsplit', generated_inputs['torch.dsplit_2'], lib="torch", suffix=2)
