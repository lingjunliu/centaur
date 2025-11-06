
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def exp2_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    inp_t = torch.tensor([0.0, 1.0, 2.5, -3.0], dtype=torch.float32)
    out_t = torch.empty_like(inp_t)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64
    inp_t = torch.tensor([[1.0, -1.0], [3.0, -4.5]], dtype=torch.float64)
    out_t = torch.zeros_like(inp_t)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0-D scalar float32
    inp_t = torch.tensor(3.5, dtype=torch.float32)
    out_t = torch.empty((), dtype=torch.float32)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32 -> out float64
    inp_t = torch.tensor([[0, 1, 2], [-2, -1, 4]], dtype=torch.int32)
    out_t = torch.empty(inp_t.shape, dtype=torch.float64)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: uint8 -> out float32
    inp_t = torch.tensor([0, 1, 8, 16, 31, 255], dtype=torch.uint8)
    out_t = torch.empty(inp_t.shape, dtype=torch.float32)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex64 1D
    inp_t = torch.tensor([1+1j, -2+0.5j, -0-3j], dtype=torch.complex64)
    out_t = torch.empty(inp_t.shape, dtype=torch.complex64)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128 3D
    real = torch.randn(2, 2, 2, dtype=torch.float64)
    imag = torch.randn(2, 2, 2, dtype=torch.float64)
    inp_t = torch.complex(real, imag)
    out_t = torch.empty_like(inp_t)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large magnitude float64
    inp_t = torch.tensor([100.0, -100.0, 709.0, -709.0], dtype=torch.float64)
    out_t = torch.empty_like(inp_t)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: empty 1D float32
    inp_t = torch.empty(0, dtype=torch.float32)
    out_t = torch.empty(0, dtype=torch.float32)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float16
    inp_t = torch.arange(-3, 3, dtype=torch.float16).reshape(1, 1, 2, 3)
    out_t = torch.empty_like(inp_t)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: int64 5D -> out float64
    inp_t = (torch.arange(12, dtype=torch.int64) - 6).reshape(1, 2, 1, 2, 3)
    out_t = torch.empty(inp_t.shape, dtype=torch.float64)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float32 with NaN/Inf
    inp_t = torch.tensor([float('nan'), float('inf'), float('-inf'), -0.0, 0.0], dtype=torch.float32)
    out_t = torch.empty_like(inp_t)
    input_dict = {"input": inp_t.numpy(), "out": out_t.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.exp2"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.exp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.exp2'.")


check_valid('torch.exp2', generated_inputs['torch.exp2'], lib="torch", suffix=0)
