
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def diagflat_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.0, -2.5, 3.3], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "offset": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([-1, 0, 2, -3], dtype=torch.int64).numpy()
    input_dict = {"input": input_arr, "offset": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([True, False, True, False], dtype=torch.bool).numpy()
    input_dict = {"input": input_arr, "offset": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([1+2j, -3+0.5j], dtype=torch.complex64).numpy()
    input_dict = {"input": input_arr, "offset": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[0.1, -0.2], [3.0, 4.5]], dtype=torch.float16).numpy()
    input_dict = {"input": input_arr, "offset": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32).numpy()
    input_dict = {"input": input_arr, "offset": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(8, dtype=torch.float32).view(2, 2, 2).numpy()
    input_dict = {"input": input_arr, "offset": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(10, dtype=torch.int64)[::2].numpy()
    input_dict = {"input": input_arr, "offset": -3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "offset": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([42], dtype=torch.int16).numpy()
    input_dict = {"input": input_arr, "offset": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[[1+0j, 2+3j], [4-1j, 0+0j]]], dtype=torch.complex128).numpy()
    input_dict = {"input": input_arr, "offset": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[255], [0], [128]], dtype=torch.uint8).numpy()
    input_dict = {"input": input_arr, "offset": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.diagflat"] = diagflat_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.diagflat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diagflat'.")


check_valid('torch.diagflat', generated_inputs['torch.diagflat'], lib="torch", suffix=0)
