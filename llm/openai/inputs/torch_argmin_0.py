
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def argmin_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, -3.5, 2.2, -3.5, 0.0], dtype=torch.float32).numpy()
    dim = 0
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.tensor([[0.5, -1.2, 3.4],
                          [7.8, 2.2, -9.0]], dtype=torch.float64).numpy()
    dim = 1
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.tensor([[5, 5, 5],
                          [5, 3, 7],
                          [8, 9, 0]], dtype=torch.int64).numpy()
    dim = -1
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.tensor([[[3, -1, 2],
                           [0, 5, -6]],
                          [[7, 4, 4],
                           [-9, 8, 1]]], dtype=torch.int32).numpy()
    dim = 0
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.tensor([[[1.5, -2.5],
                           [3.0, -1.0],
                           [0.0, 2.0]],
                          [[-1.5, 2.5],
                           [3.5, -4.0],
                           [1.0, 1.5]]], dtype=torch.float32).numpy()
    dim = 2
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    dim = 3
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.tensor([[[-10.0, 0.0, 10.0],
                           [5.0, -20.0, 15.0]],
                          [[2.5, -3.5, 4.5],
                           [6.5, 7.5, -8.5]]], dtype=torch.float32).numpy()
    dim = -2
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.arange(1 * 2 * 1 * 3 * 4, dtype=torch.int16).view(1, 2, 1, 3, 4).numpy()
    dim = 4
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.tensor([[-1, -1, -1],
                          [-1, -1, -1],
                          [2, -1, 2],
                          [2, 2, 2]], dtype=torch.int8).numpy()
    dim = 0
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.tensor([255, 0, 128, 0, 64], dtype=torch.uint8).numpy()
    dim = 0
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.tensor([[[[1.0], [2.0], [3.0]]],
                          [[[4.0], [0.0], [-1.0]]]], dtype=torch.float32).numpy()
    dim = -4
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    input = torch.tensor([[[9, -1, 0, 3, -5]],
                          [[-2, -1, -3, 4, 6]],
                          [[7, 8, -3, -4, 2]],
                          [[0, -9, 1, 5, -6]]], dtype=torch.int64).numpy()
    dim = 0
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim}))

    return list_of_inputs

generated_inputs["torch.argmin"] = argmin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.argmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argmin'.")


check_valid('torch.argmin', generated_inputs['torch.argmin'], lib="torch", suffix=0)
