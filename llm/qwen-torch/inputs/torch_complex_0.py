
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def complex_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    real = torch.tensor([1.0], dtype=torch.float64).numpy()   # tensor
    imag = torch.tensor([2.0], dtype=torch.float64).numpy()   # tensor
    out = torch.tensor([1.0], dtype=torch.complex128).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    real = torch.tensor([1.0, 2.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([3.0, 4.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([1.0, 2.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    real = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).numpy()   # tensor
    imag = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64).numpy()   # tensor
    out = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex128).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    real = torch.tensor([-1.0, -2.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([-3.0, -4.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([-1.0, -2.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    real = torch.tensor([0.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([0.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([0.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()   # tensor
    imag = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64).numpy()   # tensor
    out = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex128).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    real = torch.tensor([1.0, 2.0], dtype=torch.float64).numpy()   # tensor
    imag = torch.tensor([3.0, 4.0], dtype=torch.float64).numpy()   # tensor
    out = torch.tensor([1.0, 2.0], dtype=torch.complex128).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    real = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    real = torch.tensor([1.0, 2.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([3.0, 4.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([1.0, 2.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.complex"] = complex_inputs()

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
