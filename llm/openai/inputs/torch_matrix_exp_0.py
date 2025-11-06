
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def matrix_exp_inputs():
    list_of_inputs = []

    # Input 1: 2x2 float32 with negatives
    input = torch.tensor([[0.0, 1.0],
                          [-2.0, 3.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 2: 3x3 float64 mixed values
    input = torch.tensor([[1.5, -0.3, 0.0],
                          [2.0, 0.0, -1.0],
                          [0.7, 4.0, -5.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 3: 1x1 float32 negative
    input = torch.tensor([[-3.0]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 4: batched (5, 2, 2) float32 scaled skew-symmetric matrices
    base = torch.tensor([[0.0, -1.0],
                         [1.0,  0.0]], dtype=torch.float32)
    scales = torch.tensor([0.5, 1.0, -1.5, 2.0, -3.5], dtype=torch.float32)
    input = torch.stack([base * s for s in scales], dim=0).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 5: higher-batch (2, 3, 4, 4) float64 using arange
    input = (torch.arange(2*3*4*4, dtype=torch.float64)
                  .reshape(2, 3, 4, 4) - 50.0).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 6: 2x2 complex64
    input = torch.tensor([[1+2j, -1j],
                          [3+0j, -4+5j]], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 7: batched (2, 3, 3) complex128 constructed from real/imag parts
    r = torch.linspace(-5.0, 12.0, steps=18, dtype=torch.float64).reshape(2, 3, 3)
    im = torch.linspace(0.5, -0.5, steps=18, dtype=torch.float64).reshape(2, 3, 3)
    input = torch.complex(r, im).to(torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 8: upper-triangular 4x4 float32
    t = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    input = torch.triu(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 9: symmetric 3x3 float32
    a = torch.tensor([[0.0,  2.0, -1.0],
                      [3.0,  4.0,  5.0],
                      [-2.0, 6.0,  7.0]], dtype=torch.float32)
    input = (a + a.t()).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 10: skew-symmetric 3x3 float64
    b = torch.tensor([[0.0,  1.0,  2.0],
                      [3.0,  0.0, -4.0],
                      [-2.0, 5.0,  0.0]], dtype=torch.float64)
    input = (b - b.t()).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 11: non-contiguous via transpose 3x3 float32
    m = torch.tensor([[1.0,  2.0,  3.0],
                      [0.0, -1.0,  4.5],
                      [7.0, -2.0,  0.5]], dtype=torch.float32)
    input = m.t().numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # Input 12: very small magnitude values 2x2 float64
    input = torch.tensor([[ 1e-8, -2e-8],
                          [ 3e-8, -4e-8]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.matrix_exp"] = matrix_exp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.matrix_exp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.matrix_exp'.")


check_valid('torch.matrix_exp', generated_inputs['torch.matrix_exp'], lib="torch", suffix=0)
