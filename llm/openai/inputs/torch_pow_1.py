
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def pow_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([1.0, 2.0, 3.0, -4.0, 5.0], dtype=torch.float32).numpy()
    exponent = 2.0
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 2
    input = torch.tensor([[1.0, 4.0, 9.0],
                          [16.0, 25.0, 36.0]], dtype=torch.float64).numpy()
    exponent = 0.5
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 3
    input = torch.tensor([[[-1.0, 2.0], [3.0, -4.0]],
                          [[5.0, -6.0], [7.0, -8.0]]], dtype=torch.float32).numpy()
    exponent = 3.0
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 4 (0-dim tensor)
    input = torch.tensor(2.5, dtype=torch.float32).numpy()
    exponent = 4.0
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 5 (float16)
    input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]],
                           [[5.0, 6.0], [7.0, 8.0]]]], dtype=torch.float16).numpy()
    exponent = -1.0
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 6 (ones, exponent 1.0)
    input = torch.ones((3, 3), dtype=torch.float32).mul(10.0).numpy()
    exponent = 1.0
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 7 (includes negatives and zeros, fractional exponent)
    input = torch.tensor([[0.0, -1.0, 2.0, -3.0, 4.0, -5.0]], dtype=torch.float64).numpy()
    exponent = 2.5
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 8 (complex)
    input = torch.tensor([[1+1j, -2+0.5j],
                          [0+0j, -1-2j]], dtype=torch.complex64).numpy()
    exponent = 2.0
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 9 (empty tensor)
    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    exponent = 1.5
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 10 (non-finite values)
    input = torch.tensor([float('inf'), -float('inf'), float('nan'), 0.0, -0.0], dtype=torch.float32).numpy()
    exponent = 2.0
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 11 (5D tensor, exponent 0.0)
    input = torch.randn((1, 2, 1, 2, 3), dtype=torch.float32).numpy()
    exponent = 0.0
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    # Input 12 (zeros with negative exponent)
    input = torch.tensor([[0.0, 1.0, -2.0],
                          [3.0, 0.0, -4.0]], dtype=torch.float64).numpy()
    exponent = -2.0
    out = torch.empty_like(torch.tensor(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "exponent": exponent, "out": out}))

    return list_of_inputs

generated_inputs["torch.pow_1"] = pow_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.pow_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pow_1'.")


check_valid('torch.pow', generated_inputs['torch.pow_1'], lib="torch", suffix=1)
