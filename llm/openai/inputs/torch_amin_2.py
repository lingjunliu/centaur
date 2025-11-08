
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def amin_inputs_2():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([
        [0.6451, -0.4866, 0.2987, -1.3312],
        [-0.5744, 1.2980, 1.8397, -0.2713],
        [0.9128, 0.9214, -1.7268, -0.2995],
        [0.9023, 0.4853, 0.9075, -1.6165]
    ], dtype=torch.float32).numpy()
    dim = (1,)
    keepdim = False
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 2
    input_arr = torch.randn(3, 5, dtype=torch.float32).numpy()
    dim = (-1,)
    keepdim = True
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 3
    input_arr = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    dim = (0, 2)
    keepdim = False
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 4
    input_arr = torch.randint(-10, 10, (3, 5, 2), dtype=torch.int64).numpy()
    dim = (1,)
    keepdim = False
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 5
    input_arr = torch.randint(-1000, 1000, (2, 3, 4, 5), dtype=torch.int32).numpy()
    dim = (1, 3)
    keepdim = True
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 6
    input_arr = (torch.rand(3, 3) > 0.5).numpy()
    dim = (0, 1)
    keepdim = False
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 7
    input_arr = torch.randn(2, 1, 3, 2, 4, dtype=torch.float16).numpy()
    dim = (2,)
    keepdim = False
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 8
    input_arr = torch.randint(-128, 127, (8,), dtype=torch.int8).numpy()
    dim = (0,)
    keepdim = False
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 9
    input_arr = torch.randint(-500, 500, (2, 2, 3), dtype=torch.int16).numpy()
    dim = (-2,)
    keepdim = True
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 10
    input_arr = torch.randn(2, 4, 3, 5, dtype=torch.float32).numpy()
    dim = (1, 2)
    keepdim = False
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 11
    input_arr = torch.tensor(
        [[[1.0, float('nan')], [float('inf'), -3.0]],
         [[-2.0, 7.0], [5.0, float('-inf')]]],
        dtype=torch.float32
    ).numpy()
    dim = (2,)
    keepdim = True
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    # Input 12
    input_arr = torch.tensor([[3, -5, 0], [10, -2, 8]], dtype=torch.int64).numpy()
    dim = (0, 1)
    keepdim = True
    result = torch.amin(torch.from_numpy(input_arr), dim=dim, keepdim=keepdim)
    out = torch.empty_like(result).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "keepdim": keepdim, "out": out}))

    return list_of_inputs

generated_inputs["torch.amin_2"] = amin_inputs_2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.amin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amin_2'.")


check_valid('torch.amin', generated_inputs['torch.amin_2'], lib="torch", suffix=2)
