
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def ceil_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 with mixed signs
    x_t = torch.tensor([-0.6341, -1.4208, -1.0900, 0.5826, 2.3, -2.7], dtype=torch.float32)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 2: 2D float64 matrix
    x_t = torch.tensor([[-1.1, 0.0, 3.9], [2.2, -4.8, 5.0]], dtype=torch.float64)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 3: 0D scalar float32
    x_t = torch.tensor(3.14, dtype=torch.float32)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 4: 3D float16 tensor
    x_t = torch.randn((2, 3, 4), dtype=torch.float16)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 5: 2D int32 with negatives
    x_t = torch.tensor([[-3, 0, 7], [12, -1, -9]], dtype=torch.int32)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 6: 1D int64 with large values
    x_t = torch.tensor([1234567890123, -9876543210987, 0, -1, 1], dtype=torch.int64)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 7: 4D int8 tensor
    x_t = torch.randint(-128, 128, (2, 2, 2, 3), dtype=torch.int8)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 8: 2D uint8 tensor
    x_t = torch.tensor([[0, 1, 2, 255], [128, 200, 42, 99]], dtype=torch.uint8)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 9: 1D float32 with inf and nan
    x_t = torch.tensor([float('inf'), float('-inf'), float('nan'), -0.1, 0.0, 0.1], dtype=torch.float32)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 10: Non-contiguous 3D float32 tensor (transposed)
    x_t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2) / 3.3
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 11: Empty float32 tensor
    x_t = torch.tensor([], dtype=torch.float32)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    # Input 12: 5D float64 tensor reshaped from linspace
    x_t = torch.linspace(-5, 5, steps=32, dtype=torch.float64).reshape(1, 2, 4, 2, 2)
    input = x_t.numpy()
    out = torch.empty_like(x_t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.ceil"] = ceil_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ceil'.")


check_valid('torch.ceil', generated_inputs['torch.ceil'], lib="torch", suffix=0)
