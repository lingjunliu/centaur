
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def arccos_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 within [-1, 1]
    t = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 with values inside and outside domain
    t = torch.tensor([[-2.0, -1.0, 0.0], [0.5, 1.0, 2.0]], dtype=torch.float64)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D scalar float32
    t = torch.tensor(0.123, dtype=torch.float32)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float16 in [-1, 1]
    t = (torch.rand((2, 2, 3), dtype=torch.float16) * 2 - 1)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty 1D float32
    t = torch.empty(0, dtype=torch.float32)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Non-contiguous slice float64
    base = torch.linspace(-1, 1, steps=20, dtype=torch.float64).reshape(4, 5)
    t = base[:, ::2]
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 in [-1, 1]
    t = torch.linspace(-1, 1, steps=24, dtype=torch.float32).reshape(2, 1, 3, 4)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 1D
    t = torch.tensor([0.5 + 0.5j, -1.0 + 2.0j, 2.0 - 3.0j], dtype=torch.complex64)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex128 2x2
    t = torch.tensor([[1.0 + 0.0j, -0.5 + 1.5j], [0.0 - 2.0j, -1.0 + 0.0j]], dtype=torch.complex128)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NaNs and Infs float32
    t = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0, 1.0, -1.0], dtype=torch.float32)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: High-dim empty float64
    t = torch.empty((1, 0, 2, 3, 4), dtype=torch.float64)
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Transposed non-contiguous float32
    t = torch.linspace(-1, 1, steps=12, dtype=torch.float32).reshape(3, 4).t()
    input_dict = {"input": t.numpy(), "out": torch.empty_like(t).numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.arccos"] = arccos_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.arccos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arccos'.")


check_valid('torch.arccos', generated_inputs['torch.arccos'], lib="torch", suffix=0)
