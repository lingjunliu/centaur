
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def unbind_inputs():
    list_of_inputs = []

    input_arr = torch.arange(5).numpy()
    dim = 0
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1, 2, 3],
                              [4, 5, 6]], dtype=torch.int64).numpy()
    dim = 1
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    dim = 2
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float64).numpy()
    dim = -1
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[[True, False], [False, True]],
                              [[False, False], [True, True]]], dtype=torch.bool).numpy()
    dim = 0
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.zeros((0, 3), dtype=torch.float32).numpy()
    dim = 0
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.zeros((2, 0, 3), dtype=torch.int32).numpy()
    dim = 1
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.randn(3, 4) + 1j * torch.randn(3, 4)).to(torch.complex64).numpy()
    dim = -2
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(2 * 3 * 1 * 4, dtype=torch.float32).reshape(2, 3, 1, 4).to(torch.float16).numpy()
    dim = 2
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[[[1], [2]], [[3], [4]]],
                               [[[5], [6]], [[7], [8]]]], dtype=torch.int16).numpy()
    dim = -3
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 3, 4, 5).numpy()
    dim = 3
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 224, 224).numpy()
    dim = 0
    input_dict = {"input": input_arr, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.unbind"] = unbind_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.unbind' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unbind'.")


check_valid('torch.unbind', generated_inputs['torch.unbind'], lib="torch", suffix=0)
