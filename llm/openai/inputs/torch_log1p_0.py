
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def log1p_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 small positives
    input = torch.tensor([1e-8, 1e-6, 1e-4], dtype=torch.float32).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 2: 2D float64 with negatives near -1 and positives
    input = torch.tensor([[-0.5, -0.999999, -1.0],
                          [0.0, 1.0, 10.0]], dtype=torch.float64).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 3: 0-D scalar float32
    input = torch.tensor(0.0, dtype=torch.float32).numpy()
    out = torch.empty((), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 4: 3D float16 random in [-0.9, 0.9]
    input = torch.empty(2, 3, 4, dtype=torch.float16).uniform_(-0.9, 0.9).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 5: 4D float64 wide range magnitudes
    input = torch.logspace(-12, 12, steps=48, base=10.0, dtype=torch.float64).reshape(2, 3, 4, 2).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 6: 1D float32 non-contiguous view via slicing
    base = torch.linspace(-0.99, 3.0, steps=19, dtype=torch.float32)
    input = base[::2].numpy()
    out = torch.empty_like(base[::2]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 7: 1D complex64 values
    input = torch.tensor([(-0.5 + 0.2j), (1.0 - 2.0j), (-2.0 + 3.0j)], dtype=torch.complex64).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 8: 2D complex128 values
    input = torch.tensor([[0.0 + 0.0j, -0.5 + 0.5j],
                          [10.0 - 1.0j, -1.1 + 0.0j]], dtype=torch.complex128).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 9: 1D float32 with NaN and Infs
    input = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.5, 2.0], dtype=torch.float32).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 10: 5D float32 random in [-0.5, 0.5]
    input = (torch.rand(2, 1, 1, 3, 2, dtype=torch.float32) - 0.5).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 11: 1D float64 near the -1 boundary
    input = torch.tensor([-1.0 + 1e-12, -1.0 - 1e-12, -1.0, 0.5], dtype=torch.float64).numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 12: 2D float32 non-contiguous via transpose
    input = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    out = torch.empty_like(torch.from_numpy(input)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.log1p"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.log1p'.")


check_valid('torch.log1p', generated_inputs['torch.log1p'], lib="torch", suffix=0)
