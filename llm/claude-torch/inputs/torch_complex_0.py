
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def complex_inputs():
    list_of_inputs = []
    
    real = np.array([1.0, 2.0], dtype=np.float32)
    imag = np.array([3.0, 4.0], dtype=np.float32)
    out = np.empty(2, dtype=np.complex64)
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    imag = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    out = np.empty(3, dtype=np.complex128)
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    imag = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    out = np.empty((2, 2), dtype=np.complex64)
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = np.array([-1.0, -2.0, 3.0], dtype=np.float32)
    imag = np.array([4.0, -5.0, -6.0], dtype=np.float32)
    out = np.empty(3, dtype=np.complex64)
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    imag = np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=np.float64)
    out = np.empty((2, 2, 2), dtype=np.complex128)
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = np.array([5.0], dtype=np.float32)
    imag = np.array([10.0], dtype=np.float32)
    out = np.empty(1, dtype=np.complex64)
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    imag = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    out = np.empty(3, dtype=np.complex64)
    input_dict = {"real": real, "imag": imag, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)
    imag = np.array([[4.5, 5.5, 6.5]], dtype=np.float64)
    out = np.empty((1, 3), dtype=np.complex128)
    input_dict = {"real": real, "imag": imag, "out": out}

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.complex'.")


check_valid('torch.complex', generated_inputs['torch.complex'], lib="torch", suffix=0)
