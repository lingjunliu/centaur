
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def pow_inputs():
    list_of_inputs = []

    input = torch.tensor([0.4331, 1.2475, 0.6834, -0.2791], dtype=torch.float32).numpy()
    exponent = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float32).numpy()
    out = torch.zeros(4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.arange(1., 5., dtype=torch.float64).numpy()
    exponent = torch.tensor(3.0, dtype=torch.float64).numpy()
    out = torch.zeros(4, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=torch.float32).numpy()
    exponent = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).numpy()
    out = torch.zeros(2, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = (torch.rand(2, 1, 4, dtype=torch.float64) + 0.5).numpy()
    exponent = torch.tensor([[[1.0], [0.5], [2.0]]], dtype=torch.float64).numpy()
    out = torch.zeros(2, 3, 4, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.tensor([2, 3, 4], dtype=torch.int32).numpy()
    exponent = torch.tensor([3, 2, 1], dtype=torch.int32).numpy()
    out = torch.zeros(3, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.tensor([[10, -2],
                          [0, 5]], dtype=torch.int64).numpy()
    exponent = torch.tensor(3, dtype=torch.int64).numpy()
    out = torch.zeros(2, 2, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.tensor(2.0, dtype=torch.float32).numpy()
    exponent = torch.linspace(1, 5, steps=5, dtype=torch.float32).numpy()
    out = torch.zeros(5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.full((2, 3, 1, 1), 2.0, dtype=torch.float32).numpy()
    exponent = torch.linspace(0.5, 2.5, steps=20, dtype=torch.float32).reshape(1, 1, 4, 5).numpy()
    out = torch.zeros(2, 3, 4, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.tensor([-2.0, -3.0, -4.0], dtype=torch.float32).numpy()
    exponent = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    out = torch.zeros(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.tensor([[1.0, 2.0],
                          [4.0, 8.0]], dtype=torch.float64).numpy()
    exponent = torch.tensor([-1.0], dtype=torch.float64).numpy()
    out = torch.zeros(2, 2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.tensor([[0.5, 1.5],
                          [2.5, 3.5]], dtype=torch.float16).numpy()
    exponent = torch.tensor([[2.0, 0.5],
                             [3.0, 1.0]], dtype=torch.float16).numpy()
    out = torch.zeros(2, 2, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.arange(1., 7., dtype=torch.float32).reshape(2, 3).numpy()
    exponent = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).numpy()
    out = torch.zeros(2, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.tensor([0.0, 2.0, 0.5], dtype=torch.float32).numpy()
    exponent = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float32).numpy()
    out = torch.zeros(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    exponent = torch.ones((0, 1), dtype=torch.float32).numpy()
    out = torch.empty((0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    input = torch.tensor([-1.0, -0.1], dtype=torch.float64).numpy()
    exponent = torch.tensor([0.5, -1.5], dtype=torch.float64).numpy()
    out = torch.zeros(2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    return list_of_inputs

generated_inputs["torch.pow_2"] = pow_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.pow_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pow_2'.")


check_valid('torch.pow', generated_inputs['torch.pow_2'], lib="torch", suffix=2)
