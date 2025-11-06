
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def logsumexp_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    dim = 0
    keepdim = False
    out = torch.tensor([], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-1.0, 0.0, 1.0],
                          [2.0, -3.0, 4.0]], dtype=torch.float64).numpy()
    dim = 1
    keepdim = True
    out = torch.empty((2, 1), dtype=torch.float64).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn((3, 2, 4), dtype=torch.float16).numpy()
    dim = -1
    keepdim = False
    out = torch.empty((3, 2), dtype=torch.float16).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-1000.0, 1000.0, steps=20, dtype=torch.float32).reshape(4, 5).numpy()
    dim = 0
    keepdim = True
    out = torch.empty((1, 5), dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn((2, 3, 4, 5), dtype=torch.float32)
    t[0, 0, 0, 0] = float("inf")
    t[1, 2, 3, 4 % 5] = float("-inf")
    input = t.numpy()
    dim = 2
    keepdim = False
    out = torch.empty((2, 3, 5), dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn((2, 2, 2, 2, 2), dtype=torch.float64).numpy()
    dim = -3
    keepdim = True
    out = torch.empty((2, 2, 1, 2, 2), dtype=torch.float64).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.empty((3, 0), dtype=torch.float32).numpy()
    dim = 1
    keepdim = False
    out = torch.empty((3,), dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.empty((0, 4, 5), dtype=torch.float32).numpy()
    dim = 0
    keepdim = True
    out = torch.empty((1, 4, 5), dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn((7, 3), dtype=torch.float16).numpy()
    dim = -2
    keepdim = False
    out = torch.empty((3,), dtype=torch.float16).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.randn((2, 3, 2), dtype=torch.float32)
    t[0, 1, 1] = float("nan")
    input = t.numpy()
    dim = 2
    keepdim = True
    out = torch.empty((2, 3, 1), dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[1e-10, 1e10, -1e10],
                           [5e5, -5e5, 0.0],
                           [9.0, -9.0, 0.5],
                           [2.0, 2.0, 2.0]]], dtype=torch.float64).numpy()
    dim = 1
    keepdim = False
    out = torch.empty((1, 3), dtype=torch.float64).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([-1000.0, -1001.0, -999.0], dtype=torch.float64).numpy()
    dim = 0
    keepdim = True
    out = torch.empty((1,), dtype=torch.float64).numpy()
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.logsumexp_1"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.logsumexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.logsumexp_1'.")


check_valid('torch.special.logsumexp', generated_inputs['torch.special.logsumexp_1'], lib="torch", suffix=1)
