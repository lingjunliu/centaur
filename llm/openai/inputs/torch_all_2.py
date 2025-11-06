
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def torch_all_inputs():
    list_of_inputs = []

    input = torch.tensor([[True, False, True],
                          [True, True, True]], dtype=torch.bool).numpy()
    dim = 1
    keepdim = False
    out = torch.empty((2,), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.tensor([1, 1, 1, 0], dtype=torch.int64).numpy()
    dim = 0
    keepdim = False
    out = torch.empty((), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.tensor([[[1.0, 0.0, 2.0, 3.0],
                           [0.0, 0.0, 1.0, 1.0]],
                          [[1.0, 1.0, 1.0, 1.0],
                           [0.0, 1.0, 1.0, 0.0]],
                          [[0.0, 0.0, 0.0, 0.0],
                           [5.0, -2.0, 3.0, 0.0]]], dtype=torch.float32).numpy()
    dim = 2
    keepdim = True
    out = torch.empty((3, 2, 1), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.tensor([[1, 0],
                          [2, 3]], dtype=torch.int64).numpy()
    dim = -1
    keepdim = False
    out = torch.empty((2,), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.tensor([[1, 1, 0],
                          [1, 1, 1],
                          [0, 1, 1],
                          [1, 1, 1],
                          [1, 0, 1]], dtype=torch.uint8).numpy()
    dim = 0
    keepdim = True
    out = torch.empty((1, 3), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.ones((1, 1, 1, 1), dtype=torch.bool).numpy()
    dim = 0
    keepdim = False
    out = torch.empty((1, 1, 1), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.arange(-60, 60, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    dim = 1
    keepdim = False
    out = torch.empty((2, 4, 5), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.tensor([0, 2, 3], dtype=torch.int64).numpy()
    dim = -1
    keepdim = True
    out = torch.empty((1,), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.zeros((0, 3), dtype=torch.bool).numpy()
    dim = 0
    keepdim = False
    out = torch.empty((3,), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.zeros((0,), dtype=torch.uint8).numpy()
    dim = 0
    keepdim = False
    out = torch.empty((), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.tensor([[[1.0, -1.0, 0.0]],
                          [[2.5, 3.1, -0.2]]], dtype=torch.float32).numpy()
    dim = 1
    keepdim = True
    out = torch.empty((2, 1, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    input = torch.tensor([[[1, 0, 2],
                           [3, 4, 0],
                           [5, 6, 7]],
                          [[0, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]],
                          [[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]]], dtype=torch.int64).numpy()
    dim = 0
    keepdim = False
    out = torch.empty((3, 3), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "dim": dim, "keepdim": keepdim, "out": out}))

    return list_of_inputs

generated_inputs["torch.all_2"] = torch_all_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.all_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.all_2'.")


check_valid('torch.all', generated_inputs['torch.all_2'], lib="torch", suffix=2)
