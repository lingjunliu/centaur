
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def softmin_inputs():
    list_of_inputs = []

    # 1: 1D float32, mixed values
    input_arr = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.5, 0.0], dtype=torch.float32).numpy()
    dim = 0
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 2: 2D float32, random
    input_arr = torch.randn(2, 3, dtype=torch.float32).numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 3: 2D float32, column vector-like
    input_arr = torch.tensor([[10.0], [-5.0], [2.0], [0.1]], dtype=torch.float32).numpy()
    dim = 0
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 4: 3D float64
    input_arr = torch.linspace(-3, 3, steps=24, dtype=torch.float64).reshape(2, 3, 4).numpy()
    dim = 2
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 5: 3D float32 with large magnitude values, negative dim
    input_arr = torch.tensor([
        [[1000.0, -1000.0, 500.0, -500.0],
         [300.0, -300.0, 0.0, 1.0],
         [1e5, -1e5, 1e2, -1e2]],
        [[-1e3, 1e3, -2e2, 2e2],
         [7.0, 7.0, 7.0, 7.0],
         [-9.0, -8.0, -7.0, -6.0]]
    ], dtype=torch.float32).numpy()
    dim = -1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 6: 4D float32
    input_arr = torch.randn(2, 1, 3, 5, dtype=torch.float32).numpy()
    dim = 2
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 7: 4D float64, structured values
    input_arr = torch.arange(3*4*5*6, dtype=torch.float64).reshape(3, 4, 5, 6).numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 8: 5D float32
    input_arr = torch.randn(2, 2, 2, 2, 2, dtype=torch.float32).numpy()
    dim = 3
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 9: 2D float16 with negatives, negative dim = -2 -> 0
    input_arr = torch.tensor([
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [-3.0, 1.5, -1.5, 3.0, -2.5]
    ], dtype=torch.float16).numpy()
    dim = -2
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 10: 3D float32 with singleton dims
    input_arr = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]], dtype=torch.float32).numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 11: 1D float64 with repeated values
    input_arr = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float64).numpy()
    dim = 0
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    # 12: 2D float64 with extreme values per row
    input_arr = torch.tensor([
        [1e6, -1e6, 0.0],
        [-1e8, 1e8, 1.0],
        [100.0, 100.0, 100.0]
    ], dtype=torch.float64).numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"dim": dim, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.Softmin"] = softmin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softmin'.")


check_valid('torch.nn.Softmin', generated_inputs['torch.nn.Softmin'], lib="torch", suffix=0)
