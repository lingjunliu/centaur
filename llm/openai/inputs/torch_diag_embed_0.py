
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def diag_embed_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    offset = 0
    dim1 = -2
    dim2 = -1
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(5, dtype=torch.int64).numpy()
    offset = 2
    dim1 = 0
    dim2 = 1
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([True, False, True, True], dtype=torch.bool).numpy()
    offset = -3
    dim1 = 1
    dim2 = 0
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=torch.float64).numpy()
    offset = 1
    dim1 = 1
    dim2 = 2
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros((4, 0), dtype=torch.int32).numpy()
    offset = 0
    dim1 = -2
    dim2 = -1
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn((2, 3, 4), dtype=torch.float16).numpy()
    offset = -1
    dim1 = 0
    dim2 = 3
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = (torch.randn(1, 2, 5) + 1j * torch.randn(1, 2, 5)).to(torch.complex64).numpy()
    offset = 4
    dim1 = -3
    dim2 = -1
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(2 * 1 * 3 * 2, dtype=torch.uint8).reshape(2, 1, 3, 2).numpy()
    offset = 0
    dim1 = 2
    dim2 = -1
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[True, False, True],
                          [False, True, False]], dtype=torch.bool).numpy()
    offset = -2
    dim1 = 0
    dim2 = 2
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([], dtype=torch.float64).reshape(0).numpy()
    offset = 5
    dim1 = 0
    dim2 = 1
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(0, dtype=torch.int64).reshape(3, 0, 7).numpy()
    offset = -1
    dim1 = 2
    dim2 = 1
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2, 1, 3, 4, dtype=torch.float32).numpy()
    offset = -2
    dim1 = -6
    dim2 = -1
    input_dict = {"input": input, "offset": offset, "dim1": dim1, "dim2": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.diag_embed"] = diag_embed_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.diag_embed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diag_embed'.")


check_valid('torch.diag_embed', generated_inputs['torch.diag_embed'], lib="torch", suffix=0)
