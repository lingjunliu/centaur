
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def asinh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    t_in = torch.tensor([0.1606, -1.4267, -1.0899, -1.0250], dtype=torch.float32)
    input = t_in.numpy()
    out = torch.empty_like(t_in).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 2: 2D float64
    t_in = torch.tensor([[0.0, 1.0, -1.0],
                         [1000.0, -1000.0, 10.5]], dtype=torch.float64)
    input = t_in.numpy()
    out = torch.empty_like(t_in).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 3: 3D int32
    t_in = torch.arange(-12, 12, dtype=torch.int32).view(2, 3, 4)
    input = t_in.numpy()
    out = torch.empty(t_in.shape, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 4: 2D bool
    t_in = torch.tensor([[True, False],
                         [False, True]], dtype=torch.bool)
    input = t_in.numpy()
    out = torch.empty(t_in.shape, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 5: 1D complex64
    t_in = torch.tensor([1+2j, -3+4j, -1-1j, 0+0j], dtype=torch.complex64)
    input = t_in.numpy()
    out = torch.empty_like(t_in).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 6: 0-D scalar float32 (NaN)
    t_in = torch.tensor(float('nan'), dtype=torch.float32)
    input = t_in.numpy()
    out = torch.empty((), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 7: 4D zero-sized dimension float32
    t_in = torch.empty(2, 0, 3, 4, dtype=torch.float32)
    input = t_in.numpy()
    out = torch.empty_like(t_in).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 8: 2D non-contiguous slice float64
    base = torch.linspace(-5.0, 5.0, steps=12, dtype=torch.float64).view(3, 4)
    t_in = base[:, ::2]
    input = t_in.numpy()
    out = torch.empty_like(t_in).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 9: 1D float32 with infs
    t_in = torch.tensor([float('inf'), -float('inf'), 0.0, 1e20], dtype=torch.float32)
    input = t_in.numpy()
    out = torch.empty_like(t_in).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 10: 0-D scalar int64
    t_in = torch.tensor(-5, dtype=torch.int64)
    input = t_in.numpy()
    out = torch.empty((), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 11: 2D complex128
    t_in = torch.tensor([[1-1j, 2+0j],
                         [0-3j, -4+5j]], dtype=torch.complex128)
    input = t_in.numpy()
    out = torch.empty_like(t_in).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 12: 1D float32 small magnitudes
    t_in = torch.tensor([-0.0, 1e-8, -1e-8, 3.14159], dtype=torch.float32)
    input = t_in.numpy()
    out = torch.empty_like(t_in).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.asinh"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.asinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asinh'.")


check_valid('torch.asinh', generated_inputs['torch.asinh'], lib="torch", suffix=0)
