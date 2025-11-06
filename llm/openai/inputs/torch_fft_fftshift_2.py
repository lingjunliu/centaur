
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def fftshift_2_inputs():
    list_of_inputs = []
    
    # Input 1: 1D even length, float64
    arr = torch.arange(8, dtype=torch.float64).numpy()
    dim = (0,)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 2: 1D odd length, complex64
    real = torch.randn(9, dtype=torch.float32)
    imag = torch.randn(9, dtype=torch.float32)
    arr = (real + 1j * imag).numpy()
    dim = (0,)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 3: 2D square, complex128, shift along both dims
    real = torch.randn(4, 4, dtype=torch.float64)
    imag = torch.randn(4, 4, dtype=torch.float64)
    arr = (real + 1j * imag).numpy()
    dim = (0, 1)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 4: 2D non-square, float32, shift along columns only
    arr = torch.linspace(0, 1, steps=40, dtype=torch.float32).reshape(5, 8).numpy()
    dim = (1,)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 5: 3D, int32, shift along last dim with negative index
    arr = torch.randint(-50, 50, (3, 4, 5), dtype=torch.int32).numpy()
    dim = (-1,)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 6: 3D with singleton dim, float16, shift along first and last dims
    arr = torch.randn(6, 1, 7, dtype=torch.float16).numpy()
    dim = (0, 2)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 7: 4D, complex64, shift along non-consecutive dims (unsorted)
    real = torch.randn(2, 3, 4, 5, dtype=torch.float32)
    imag = torch.randn(2, 3, 4, 5, dtype=torch.float32)
    arr = (real + 1j * imag).numpy()
    dim = (3, 1)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 8: 2D even-even, int64, shift along both dims with negative indices
    arr = torch.arange(36, dtype=torch.int64).reshape(6, 6).numpy()
    dim = (-2, -1)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 9: 5D, bool, shift along all dims
    arr = (torch.rand(2, 2, 2, 2, 2) > 0.5).numpy()
    dim = (0, 1, 2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 10: 2D with first dim 1, float32, shift along both dims
    arr = torch.arange(10, dtype=torch.float32).reshape(1, 10).numpy()
    dim = (0, 1)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 11: 3D cube, complex64, shift along all dims
    real = torch.randn(8, 8, 8, dtype=torch.float32)
    imag = torch.randn(8, 8, 8, dtype=torch.float32)
    arr = (real + 1j * imag).numpy()
    dim = (0, 1, 2)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    # Input 12: 1D even length, int16, shift with negative dim index
    arr = torch.arange(-12, 12, 3, dtype=torch.int16).numpy()
    dim = (-1,)
    list_of_inputs.append(copy.deepcopy({"input": arr, "dim": dim}))
    
    return list_of_inputs

generated_inputs["torch.fft.fftshift_2"] = fftshift_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.fftshift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fftshift_2'.")


check_valid('torch.fft.fftshift', generated_inputs['torch.fft.fftshift_2'], lib="torch", suffix=2)
