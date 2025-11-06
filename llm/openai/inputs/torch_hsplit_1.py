
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def hsplit_inputs():
    list_of_inputs = []

    input_arr = torch.arange(6., dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(-8, 0, dtype=torch.int64).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(12., dtype=torch.float32).reshape(2, 6).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.arange(16, dtype=torch.int32).reshape(4, 4) - 8).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(-4.5, 4.5, steps=9, dtype=torch.float64).reshape(1, 9).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 6, 3, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 8, 2, 2, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(0, 2, (5, 10), dtype=torch.int8).bool().numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 6, dtype=torch.complex64).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.empty(4, 0, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.empty(0, 6, 2, dtype=torch.int64).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(7, 14, dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 9, 1, 1, 1, dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(36, dtype=torch.int16).reshape(3, 12).numpy()
    input_dict = {"input": input_arr, "indices_or_sections": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.hsplit_1"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.hsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hsplit_1'.")


check_valid('torch.hsplit', generated_inputs['torch.hsplit_1'], lib="torch", suffix=1)
