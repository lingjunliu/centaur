
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def diagonal_inputs():
    list_of_inputs = []

    input = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "offset": 0, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 5, dtype=torch.float64).numpy()
    input_dict = {"input": input, "offset": 1, "dim1": 1, "dim2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(2 * 3 * 3, dtype=torch.int64).reshape(2, 3, 3).numpy()
    input_dict = {"input": input, "offset": 0, "dim1": -2, "dim2": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {"input": input, "offset": -1, "dim1": 1, "dim2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 3, dtype=torch.float32).numpy()
    input_dict = {"input": input, "offset": -2, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = (torch.randint(0, 2, (3, 4, 4)) > 0).numpy()
    input_dict = {"input": input, "offset": 2, "dim1": 1, "dim2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = (torch.randn(2, 2, 3, 3) + 1j * torch.randn(2, 2, 3, 3)).numpy()
    input_dict = {"input": input, "offset": 0, "dim1": -2, "dim2": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 3, dtype=torch.float64).numpy()
    input_dict = {"input": input, "offset": 5, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 5, 4, 2, dtype=torch.float64).numpy()
    input_dict = {"input": input, "offset": -1, "dim1": 1, "dim2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 4, 5, dtype=torch.float32).numpy()
    input_dict = {"input": input, "offset": 0, "dim1": 0, "dim2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(6 * 4 * 3, dtype=torch.int32).reshape(6, 4, 3).numpy()
    input_dict = {"input": input, "offset": 1, "dim1": 0, "dim2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[1.0]], dtype=torch.float16).numpy()
    input_dict = {"input": input, "offset": 0, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 3, 5).numpy()
    input_dict = {"input": input, "offset": -1, "dim1": -3, "dim2": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict = {"input": input, "offset": 0, "dim1": 2, "dim2": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.diagonal"] = diagonal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.diagonal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diagonal'.")


check_valid('torch.diagonal', generated_inputs['torch.diagonal'], lib="torch", suffix=0)
