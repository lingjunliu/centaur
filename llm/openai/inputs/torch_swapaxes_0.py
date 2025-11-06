
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def swapaxes_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, swapping the only axis with itself
    input = torch.arange(5, dtype=torch.float32).numpy()
    axis0 = 0
    axis1 = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 2: 2D int64, swap rows and columns
    input = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64).numpy()
    axis0 = 0
    axis1 = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 3: 2D complex64, using negative axes
    real = torch.randn(3, 4, dtype=torch.float32)
    imag = torch.randn(3, 4, dtype=torch.float32)
    input = torch.complex(real, imag).numpy()
    axis0 = -2
    axis1 = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 4: 3D float32, swap first and last axes
    input = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4).numpy()
    axis0 = 0
    axis1 = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 5: 3D bool, swap last and first using negative indices
    input = (torch.rand(2, 3, 3) > 0.5).numpy()
    axis0 = -1
    axis1 = -3
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 6: 3D int8 with a zero-sized dimension, swap middle and last axes
    input = torch.zeros((2, 0, 4), dtype=torch.int8).numpy()
    axis0 = 1
    axis1 = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 7: 4D float16, swap channel and last dimension
    input = torch.randn(2, 3, 4, 5, dtype=torch.float16).numpy()
    axis0 = 1
    axis1 = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 8: 4D uint8, swap batch and depth dimensions
    input = torch.arange(2 * 2 * 2 * 2, dtype=torch.uint8).reshape(2, 2, 2, 2).numpy()
    axis0 = 0
    axis1 = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 9: 5D float64, swap first and second-last dimensions
    input = torch.randn(1, 2, 3, 4, 5, dtype=torch.float64).numpy()
    axis0 = 0
    axis1 = -2
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 10: 5D complex128, swap last two dimensions
    real = torch.randn(2, 1, 3, 4, 5, dtype=torch.float64)
    imag = torch.randn(2, 1, 3, 4, 5, dtype=torch.float64)
    input = torch.complex(real, imag).numpy()
    axis0 = 3
    axis1 = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 11: 2D int16, same axes (no-op)
    input = torch.tensor([[7, 8], [9, 10]], dtype=torch.int16).numpy()
    axis0 = -1
    axis1 = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    # Input 12: 3D int32, swap middle and first dimensions using mixed signs
    input = torch.arange(2 * 3 * 4, dtype=torch.int32).reshape(2, 3, 4).numpy()
    axis0 = 1
    axis1 = -3
    list_of_inputs.append(copy.deepcopy({"input": input, "axis0": axis0, "axis1": axis1}))

    return list_of_inputs

generated_inputs["torch.swapaxes"] = swapaxes_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.swapaxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.swapaxes'.")


check_valid('torch.swapaxes', generated_inputs['torch.swapaxes'], lib="torch", suffix=0)
