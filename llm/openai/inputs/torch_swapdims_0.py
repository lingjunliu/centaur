
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def swapdims_inputs():
    list_of_inputs = []

    # 1D float32, swapping same dim (no-op)
    input = torch.arange(5, dtype=torch.float32).numpy()
    dim0 = 0
    dim1 = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 2D int64, swap 0 and 1
    input = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64).numpy()
    dim0 = 0
    dim1 = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 2D float64, swap 1 and 0
    input = torch.randn(3, 4, dtype=torch.float64).numpy()
    dim0 = 1
    dim1 = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 3D float32, swap first and last dims
    input = torch.arange(2*3*4, dtype=torch.float32).reshape(2, 3, 4).numpy()
    dim0 = 0
    dim1 = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 3D int16, use negative dims
    input = torch.arange(24, dtype=torch.int16).reshape(2, 3, 4).numpy()
    dim0 = -1
    dim1 = -3
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 4D bool, swap inner and last dims
    input = torch.tensor([[[[True, False, True, False]],
                           [[False, True, False, True]],
                           [[True, True, False, False]]],
                          [[[False, False, True, True]],
                           [[True, False, True, False]],
                           [[False, True, True, False]]]], dtype=torch.bool).numpy()
    dim0 = 1
    dim1 = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 5D float16, swap -2 and 0
    input = torch.randn(1, 2, 3, 4, 5, dtype=torch.float16).numpy()
    dim0 = -2
    dim1 = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 3D int32 with a zero-sized dimension, swap dim 1 and 2
    input = torch.empty((2, 0, 5), dtype=torch.int32).numpy()
    dim0 = 1
    dim1 = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 6D uint8, swap dims 2 and 4
    input = torch.arange(2*3*1*4*1*5, dtype=torch.uint8).reshape(2, 3, 1, 4, 1, 5).numpy()
    dim0 = 2
    dim1 = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 3D complex64, swap 0 and 1
    input = (torch.randn(2, 2, 2) + 1j * torch.randn(2, 2, 2)).numpy()
    dim0 = 0
    dim1 = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 3D complex128, swap -2 and -1
    input = (torch.randn(1, 2, 3, dtype=torch.float64) + 1j * torch.randn(1, 2, 3, dtype=torch.float64)).numpy()
    dim0 = -2
    dim1 = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    # 4D int8 with a zero-sized leading dimension, swap 0 and 2
    input = torch.empty((0, 2, 3, 4), dtype=torch.int8).numpy()
    dim0 = 0
    dim1 = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "dim0": dim0, "dim1": dim1}))

    return list_of_inputs

generated_inputs["torch.swapdims"] = swapdims_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.swapdims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.swapdims'.")


check_valid('torch.swapdims', generated_inputs['torch.swapdims'], lib="torch", suffix=0)
