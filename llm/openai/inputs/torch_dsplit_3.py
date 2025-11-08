
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def dsplit_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(16.0, dtype=torch.float32).reshape(2, 2, 4).numpy()
    indices_or_sections = (2,)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.arange(3 * 3 * 6, dtype=torch.int64).reshape(3, 3, 6).numpy()
    indices_or_sections = (1, 4)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.linspace(-10, 10, steps=20, dtype=torch.float64).reshape(4, 1, 5).numpy()
    indices_or_sections = (0, 5)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.empty((2, 3, 0), dtype=torch.float32).numpy()
    indices_or_sections = (0,)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randint(0, 2, (2, 2, 7)).bool().numpy()
    indices_or_sections = (3,)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.randn(2, 2, 4, 3, dtype=torch.float16).numpy()
    indices_or_sections = (1, 3)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.arange(1 * 5 * 9 * 2 * 2, dtype=torch.int32).reshape(1, 5, 9, 2, 2).numpy()
    indices_or_sections = (2, 5, 9)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.randint(0, 256, (3, 2, 3), dtype=torch.uint8).numpy()
    indices_or_sections = (1, 2)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    real = torch.randn(2, 4, 8, dtype=torch.float32)
    imag = torch.randn(2, 4, 8, dtype=torch.float32)
    input_arr = (real + 1j * imag).numpy()
    indices_or_sections = (2, 5)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.ones((2, 2, 1), dtype=torch.float32).numpy()
    indices_or_sections = (0,)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.arange(6 * 1 * 10, dtype=torch.int16).reshape(6, 1, 10).numpy()
    indices_or_sections = (0, 3, 10)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn(2, 3, 6, dtype=torch.float32).numpy()
    indices_or_sections = (-3, -1)
    input_dict = {"input": input_arr, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dsplit_3"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dsplit_3'.")


check_valid('torch.dsplit', generated_inputs['torch.dsplit_3'], lib="torch", suffix=3)
