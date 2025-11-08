
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def tanhshrink_inputs():
    list_of_inputs = []

    # 1: 1D float32 with negative, zero, positive
    input = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 2: 2D float64 matrix
    input = torch.tensor([[1.5, -2.2, 0.0], [3.3, -4.4, 5.5]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 3: 0D scalar float32
    input = torch.tensor(0.5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 4: 3D float16 tensor
    input = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 5: 4D float32 tensor (e.g., NCHW)
    input = torch.randn(1, 3, 8, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 6: 5D float32 tensor
    input = torch.arange(0, 12, dtype=torch.float32).view(2, 1, 2, 1, 3).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 7: Non-contiguous via transpose (float32)
    t = torch.arange(12.0, dtype=torch.float32).view(3, 4).t()
    input = t.numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 8: Non-contiguous via slicing step (float32)
    input = torch.linspace(-5, 5, steps=20, dtype=torch.float32)[::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 9: Empty dimension (0, 5)
    input = torch.zeros((0, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 10: Wide range magnitudes float64
    input = torch.tensor([1e-8, -1e-6, 1e-3, -1.0, 10.0, -100.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 11: Values with inf and nan (float32)
    input = torch.tensor([float('inf'), float('-inf'), float('nan'), 0.0, 2.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 12: Complex64 1D
    input = torch.tensor([1+1j, -2+0.5j, 0-1j, 3+0j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 13: Complex128 2D
    input = (torch.randn(2, 3, dtype=torch.float64) + 1j * torch.randn(2, 3, dtype=torch.float64)).to(torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    # 14: Broadcast-like pattern but materialized (repeat) float32
    base = torch.tensor([[-0.1, 0.0, 0.1]], dtype=torch.float32)
    input = base.repeat(4, 3).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.nn.functional.tanhshrink"] = tanhshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.tanhshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.tanhshrink'.")


check_valid('torch.nn.functional.tanhshrink', generated_inputs['torch.nn.functional.tanhshrink'], lib="torch", suffix=0)
