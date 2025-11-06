
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def addmv_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    mat = torch.tensor([[0.1, -0.2, 0.3, 1.0],
                        [1.5, 2.0, -0.5, -1.0],
                        [0.0, 0.25, 0.75, -0.75]], dtype=torch.float32).numpy()
    vec = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32).numpy()
    beta = 1.0
    alpha = 1.0
    out = torch.zeros(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 2
    input_arr = torch.tensor(2.5, dtype=torch.float64).numpy()
    mat = torch.tensor([[1.0, 2.0],
                        [3.0, 4.0]], dtype=torch.float64).numpy()
    vec = torch.tensor([-1.0, 0.5], dtype=torch.float64).numpy()
    beta = 0.5
    alpha = 2.0
    out = torch.zeros(2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 3
    input_arr = torch.tensor([-5.0, 5.0], dtype=torch.float32).numpy()
    mat = torch.tensor([[2.0, -1.0],
                        [0.0, 3.0]], dtype=torch.float32).numpy()
    vec = torch.tensor([4.0, -2.0], dtype=torch.float32).numpy()
    beta = 0.0
    alpha = -1.0
    out = torch.zeros(2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 4
    input_arr = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float16).numpy()
    mat = torch.tensor([[1.0, 0.0, -1.0],
                        [0.5, -0.5, 1.5],
                        [-1.0, 2.0, 0.0],
                        [0.25, 0.75, -0.25]], dtype=torch.float16).numpy()
    vec = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float16).numpy()
    beta = 1.0
    alpha = 0.25
    out = torch.zeros(4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 5
    input_arr = torch.tensor([10.0], dtype=torch.float64).numpy()
    mat = torch.tensor([[1.0, 2.0, -3.0, 4.0, -5.0]], dtype=torch.float64).numpy()
    vec = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5], dtype=torch.float64).numpy()
    beta = -2.0
    alpha = 3.0
    out = torch.zeros(1, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 6
    input_arr = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5], dtype=torch.float32).numpy()
    mat = torch.tensor([[1.0], [2.0], [-3.0], [4.0], [-5.0]], dtype=torch.float32).numpy()
    vec = torch.tensor([2.0], dtype=torch.float32).numpy()
    beta = 2.0
    alpha = 1.5
    out = torch.zeros(5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 7
    input_arr = torch.empty(0, dtype=torch.float32).numpy()
    mat = torch.empty((0, 0), dtype=torch.float32).numpy()
    vec = torch.empty(0, dtype=torch.float32).numpy()
    beta = 1.0
    alpha = 1.0
    out = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 8
    input_arr = torch.tensor([-1e3, 2e3, -3e3], dtype=torch.float32).numpy()
    mat = torch.tensor([[100.0, 200.0, 300.0],
                        [-400.0, 500.0, -600.0],
                        [700.0, -800.0, 900.0]], dtype=torch.float32).numpy()
    vec = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32).numpy()
    beta = 1.0
    alpha = 10.0
    out = torch.zeros(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 9
    input_arr = torch.tensor([0.25], dtype=torch.float32).numpy()
    mat = torch.eye(7, dtype=torch.float32).numpy()
    vec = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float32).numpy()
    beta = 1.0
    alpha = 0.0
    out = torch.zeros(7, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 10
    input_arr = torch.tensor([-10.0, 10.0], dtype=torch.float64).numpy()
    mat = torch.tensor([[-1.0, 2.0, 3.0],
                        [4.0, -5.0, 6.0]], dtype=torch.float64).numpy()
    vec = torch.tensor([1.5, -2.5, 0.5], dtype=torch.float64).numpy()
    beta = 0.75
    alpha = -2.5
    out = torch.zeros(2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 11
    input_arr = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5], dtype=torch.float16).numpy()
    mat = torch.tensor([[1.0, -1.0],
                        [0.5, 0.5],
                        [-0.5, 1.5],
                        [2.0, -2.0],
                        [0.25, -0.75],
                        [-1.25, 0.25]], dtype=torch.float16).numpy()
    vec = torch.tensor([2.0, -3.0], dtype=torch.float16).numpy()
    beta = 1.0
    alpha = 1.0
    out = torch.zeros(6, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    # 12
    input_arr = torch.tensor(0.0, dtype=torch.float32).numpy()
    mat = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0, 0.0],
                        [0.0, 0.0, 0.0, 4.0]], dtype=torch.float32).numpy()
    vec = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy()
    beta = 0.0
    alpha = 1.0
    out = torch.zeros(4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": input_arr, "mat": mat, "vec": vec, "beta": beta, "alpha": alpha, "out": out
    }))

    return list_of_inputs

generated_inputs["torch.addmv"] = addmv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addmv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv'.")


check_valid('torch.addmv', generated_inputs['torch.addmv'], lib="torch", suffix=0)
