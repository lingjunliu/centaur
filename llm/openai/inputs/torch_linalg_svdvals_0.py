
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def svdvals_inputs():
    list_of_inputs = []

    def out_dtype_for(dtype):
        if dtype == torch.complex64:
            return torch.float32
        if dtype == torch.complex128:
            return torch.float64
        return dtype

    # Input 1: 2D square, float32
    A_t = torch.randn(3, 3, dtype=torch.float32)
    out_t = torch.empty(3, dtype=out_dtype_for(A_t.dtype))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 2: Tall matrix, float64
    A_t = torch.randn(5, 3, dtype=torch.float64) * 2 - 1
    out_t = torch.empty(3, dtype=out_dtype_for(A_t.dtype))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 3: Wide matrix, float32
    A_t = torch.randn(3, 5, dtype=torch.float32) * 10
    out_t = torch.empty(3, dtype=out_dtype_for(A_t.dtype))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 4: Batch of square matrices, float32
    A_t = torch.randn(2, 4, 4, dtype=torch.float32)
    out_t = torch.empty(2, 4, dtype=out_dtype_for(A_t.dtype))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 5: Batch of rectangular matrices, float64
    A_t = torch.randn(2, 3, 2, dtype=torch.float64)
    out_t = torch.empty(2, 2, dtype=out_dtype_for(A_t.dtype))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 6: 4D batch, float32
    A_t = torch.randn(2, 3, 6, 4, dtype=torch.float32)
    out_t = torch.empty(2, 3, 4, dtype=out_dtype_for(A_t.dtype))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 7: Complex64 rectangular
    A_t = (torch.randn(4, 3, dtype=torch.float32) +
           1j * torch.randn(4, 3, dtype=torch.float32))
    out_t = torch.empty(3, dtype=out_dtype_for(torch.complex64))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 8: Complex128 square
    A_t = (torch.randn(5, 5, dtype=torch.float64) +
           1j * torch.randn(5, 5, dtype=torch.float64))
    out_t = torch.empty(5, dtype=out_dtype_for(torch.complex128))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 9: Batch complex64 square
    A_t = (torch.randn(4, 3, 3, dtype=torch.float32) +
           1j * torch.randn(4, 3, 3, dtype=torch.float32))
    out_t = torch.empty(4, 3, dtype=out_dtype_for(torch.complex64))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 10: Non-contiguous slice, float64
    base = torch.arange(6 * 4, dtype=torch.float64).reshape(6, 4)
    A_t = base[::2, ::2]  # shape (3, 2)
    out_t = torch.empty(2, dtype=out_dtype_for(A_t.dtype))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 11: Batch with structured values, float32
    A_t = torch.arange(3 * 5 * 3, dtype=torch.float32).reshape(3, 5, 3) - 50.0
    out_t = torch.empty(3, 3, dtype=out_dtype_for(A_t.dtype))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    # Input 12: 1x1 matrix, float64
    A_t = torch.tensor([[-5.0]], dtype=torch.float64)
    out_t = torch.empty(1, dtype=out_dtype_for(A_t.dtype))
    list_of_inputs.append(copy.deepcopy({
        "A": A_t.numpy(),
        "driver": None,
        "out": out_t.numpy()
    }))

    return list_of_inputs

generated_inputs["torch.linalg.svdvals"] = svdvals_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.svdvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.svdvals'.")


check_valid('torch.linalg.svdvals', generated_inputs['torch.linalg.svdvals'], lib="torch", suffix=0)
