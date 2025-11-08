
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def reciprocal_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    inp_t = torch.tensor([1.0, -2.0, 0.5, -0.25], dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 with zero
    inp_t = torch.tensor([[1.0, -4.0, 0.0],
                          [2.5, -0.1, 10.0]], dtype=torch.float64)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 promoted to float, out float32
    inp_t = torch.tensor([1, -2, 3, -4, 0], dtype=torch.int32)
    input = inp_t.numpy()
    out = torch.empty(inp_t.shape, dtype=torch.float32).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64, out float32
    inp_t = torch.arange(-3, 3, dtype=torch.int64).reshape(2, 3)
    input = inp_t.numpy()
    out = torch.empty(inp_t.shape, dtype=torch.float32).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D uint8, out float32
    inp_t = torch.tensor([[[1, 2],
                           [0, 255]]], dtype=torch.uint8)
    input = inp_t.numpy()
    out = torch.empty(inp_t.shape, dtype=torch.float32).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0-D scalar float32
    inp_t = torch.tensor(-3.5, dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty((), dtype=torch.float32).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Non-contiguous slice, float32
    base = torch.linspace(1, 12, steps=12, dtype=torch.float32).reshape(3, 4)
    inp_t = base[:, ::2]
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 with an explicit zero
    inp_t = torch.randn(2, 1, 3, 4, dtype=torch.float32)
    inp_t[0, 0, 0, 0] = 0.0
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large magnitude values, float64
    inp_t = torch.tensor([1e-8, 1e8, -1e-12, -1e12], dtype=torch.float64)
    input = inp_t.numpy()
    out = torch.empty_like(inp_t).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int16, out float32
    inp_t = torch.arange(-6, 2, dtype=torch.int16).reshape(2, 2, 2)
    input = inp_t.numpy()
    out = torch.empty(inp_t.shape, dtype=torch.float32).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: out aliases input (float32)
    inp_t = torch.tensor([[0.5, -0.25],
                          [2.0, -1.0]], dtype=torch.float32)
    input = inp_t.numpy()
    out = input
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty tensor (0x3), float32
    inp_t = torch.empty((0, 3), dtype=torch.float32)
    input = inp_t.numpy()
    out = torch.empty((0, 3), dtype=torch.float32).numpy()
    input_dict = {"input": input, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.reciprocal"] = reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.reciprocal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.reciprocal'.")


check_valid('torch.reciprocal', generated_inputs['torch.reciprocal'], lib="torch", suffix=0)
