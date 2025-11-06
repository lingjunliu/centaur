
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def elu_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).numpy()
    alpha = np.float32(1.0)
    inplace = np.bool_(False)
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.linspace(-3, 3, steps=6, dtype=torch.float64).reshape(2, 3).numpy()
    alpha = np.float64(0.5)
    inplace = np.bool_(True)
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    alpha = float(1.3)
    inplace = False
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = (torch.arange(24, dtype=torch.float16).reshape(2, 3, 4) - 12).numpy()
    alpha = np.float16(1.0)
    inplace = np.bool_(False)
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.tensor([[-1000.0, -5.0, 5.0, 1000.0]], dtype=torch.float32).numpy()
    alpha = np.float64(2.0)
    inplace = True
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.zeros((0, 3), dtype=torch.float32).numpy()
    alpha = np.float32(1.0)
    inplace = False
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.randn(1, 3, 8, 8, dtype=torch.float32).numpy()
    alpha = 0.01
    inplace = False
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.randn(4, 5, dtype=torch.float64).t().numpy()
    alpha = 3.0
    inplace = True
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.tensor([[np.nan, np.inf, -np.inf, -1.0, 0.0, 1.0]], dtype=torch.float64).numpy()
    alpha = 1.0
    inplace = False
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.randn(2, 2, 2, 2, dtype=torch.float32).numpy()
    alpha = 0.5
    inplace = True
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.linspace(-0.1, 0.1, steps=10, dtype=torch.float32).reshape(5, 2).numpy()
    alpha = -0.5
    inplace = False
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = (torch.rand(3, dtype=torch.float32) * 20 - 10).numpy()
    alpha = 5.0
    inplace = True
    input_dict = {
        "input": input_arr,
        "alpha": alpha,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.elu"] = elu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.elu'.")


check_valid('torch.nn.functional.elu', generated_inputs['torch.nn.functional.elu'], lib="torch", suffix=0)
