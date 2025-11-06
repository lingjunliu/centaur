
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def matrix_power_inputs():
    list_of_inputs = []

    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).numpy()
    n = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.tensor([[1, 0, 2], [0, -1, 0], [3, 0, 1]], dtype=torch.int64).numpy()
    n = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.tensor([[-2.0]], dtype=torch.float64).numpy()
    n = 5
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.tensor([[2.0, 0.0], [0.0, 5.0]], dtype=torch.float64).numpy()
    n = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.diag(torch.tensor([2.0, 0.5], dtype=torch.float32)).numpy()
    n = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.diag(torch.tensor([1+1j, 2-1j, -0.5+0.3j], dtype=torch.complex64)).numpy()
    n = -2
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.stack([
        torch.tensor([[1, 1], [0, 1]], dtype=torch.int32),
        torch.tensor([[2, 0], [0, 2]], dtype=torch.int32),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        torch.tensor([[-1, 2], [3, -2]], dtype=torch.int32),
    ], dim=0).numpy()
    n = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.stack([
        torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        torch.diag(torch.tensor([2.0, 1.0, 3.0], dtype=torch.float32)),
        torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
    ], dim=0).numpy()
    n = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    eye3 = torch.eye(3, dtype=torch.float64)
    input = torch.stack([
        torch.stack([eye3 * 1.5, eye3 * 2.0], dim=0),
        torch.stack([eye3 * 3.0, eye3 * 0.75], dim=0),
    ], dim=0).numpy()
    n = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.tensor([[1, 2], [-3, 4]], dtype=torch.int8).numpy()
    n = 7
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.tensor([
        [0.0, -1.0, 2.0, 0.0, 3.0],
        [4.0, 5.0, 0.0, -2.0, 1.0],
        [-1.0, 2.0, 3.0, 4.0, -2.0],
        [0.0, -3.0, 1.0, 2.0, 0.0],
        [2.0, 0.0, -1.0, 0.0, 1.0]
    ], dtype=torch.float32).numpy()
    n = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.diag(torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)).numpy()
    n = -3
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    input = torch.diag_embed(torch.tensor([[1.5, 2.0, 2.5, 3.0],
                                           [4.0, 0.5, 1.0, 1.25]], dtype=torch.float64)).numpy()
    n = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "n": n}))

    return list_of_inputs

generated_inputs["torch.matrix_power"] = matrix_power_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.matrix_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.matrix_power'.")


check_valid('torch.matrix_power', generated_inputs['torch.matrix_power'], lib="torch", suffix=0)
