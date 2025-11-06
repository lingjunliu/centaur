
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def celu_inputs():
    list_of_inputs = []
    
    # Input 1
    inp = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    inp = torch.ones((2, 3), dtype=torch.float64).numpy()
    input_dict = {
        "input": inp,
        "alpha": 0.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    inp = torch.randn((2, 3, 4), dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "alpha": 1.5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    inp = torch.tensor(-1.5, dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "alpha": 2.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    inp = torch.empty((0,), dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    inp = torch.randn((2, 3, 4, 5), dtype=torch.float16).numpy()
    input_dict = {
        "input": inp,
        "alpha": 0.1,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 (non-contiguous slice)
    base = torch.arange(25, dtype=torch.float64).reshape(5, 5)
    inp = base[::2, 1::2].numpy()
    input_dict = {
        "input": inp,
        "alpha": 1.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 (NaN and Inf)
    inp = torch.tensor([float('-inf'), -1.0, 0.0, 1.0, float('inf'), float('nan')], dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 (wide range)
    inp = torch.linspace(-50.0, 50.0, steps=101, dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "alpha": 0.5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 (empty middle dimension)
    inp = torch.empty((2, 0, 3), dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "alpha": 0.9,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 (5D tensor)
    inp = torch.zeros((1, 2, 3, 1, 4), dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "alpha": 3.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12 (very small alpha)
    inp = torch.tensor([-1e-3, -1e-6, 0.0, 1e-6, 1e-3], dtype=torch.float64).numpy()
    input_dict = {
        "input": inp,
        "alpha": 1e-6,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13 (random negative/positive vector)
    inp = torch.randn(4, dtype=torch.float32).numpy()
    input_dict = {
        "input": inp,
        "alpha": 10.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.celu'.")


check_valid('torch.nn.functional.celu', generated_inputs['torch.nn.functional.celu'], lib="torch", suffix=0)
