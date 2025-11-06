
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ifftshift_inputs_2():
    list_of_inputs = []
    
    # Input 1: 1D float64
    input_arr = np.arange(5, dtype=np.float64)
    dim = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 2: 1D int64 with negatives and evens
    input_arr = np.arange(-6, 6, 2, dtype=np.int64)
    dim = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 3: 2D float32
    input_arr = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)
    dim = (0, 1)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 4: 2D complex64, shift last dim only
    a = np.arange(20, dtype=np.float32).reshape(4, 5)
    b = (a * 0.5).astype(np.float32)
    input_arr = (a + 1j * b).astype(np.complex64)
    dim = (-1,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 5: 3D int32, shift first and last dims
    input_arr = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
    dim = (0, 2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 6: 3D complex128, shift all dims
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    b = -np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    input_arr = (a + 1j * b).astype(np.complex128)
    dim = (0, 1, 2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 7: 4D bool, shift subset dims
    input_arr = (np.arange(120).reshape(2, 3, 4, 5) % 2 == 0)
    dim = (1, 3)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 8: 4D float16 with singleton dim, negative dims
    input_arr = np.linspace(-3, 3, 24, dtype=np.float32).reshape(3, 1, 2, 4).astype(np.float16)
    dim = (-4, -2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 9: zero-length 1D float32
    input_arr = np.array([], dtype=np.float32)
    dim = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 10: 2D float64 non-contiguous view
    base = np.arange(60, dtype=np.float64).reshape(5, 12)
    input_arr = base[:, ::3]
    dim = (0, 1)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 11: 5D int16, shift non-adjacent dims
    input_arr = np.arange(32, dtype=np.int16).reshape(2, 2, 2, 2, 2)
    dim = (0, 2, 4)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    # Input 12: 2D float32, shift last dim only
    input_arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    dim = (1,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim}))
    
    return list_of_inputs

generated_inputs["torch.fft.ifftshift_2"] = ifftshift_inputs_2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.ifftshift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifftshift_2'.")


check_valid('torch.fft.ifftshift', generated_inputs['torch.fft.ifftshift_2'], lib="torch", suffix=2)
