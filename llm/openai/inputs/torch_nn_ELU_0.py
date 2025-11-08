
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def elu_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.tensor([-1.5, -0.5, 0.0, 0.5, 2.3], dtype=torch.float32).numpy()
    input_dict = {
        "alpha": 1.0,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(3, 4, dtype=torch.float64).numpy()
    input_dict = {
        "alpha": 0.5,
        "inplace": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    input_dict = {
        "alpha": 1.2,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = np.array(-0.3, dtype=np.float32)
    input_dict = {
        "alpha": 1.0,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.randn(2, 1, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "alpha": 2.0,
        "inplace": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = torch.randn(1, 2, 1, 2, 3, dtype=torch.float16).numpy()
    input_dict = {
        "alpha": 0.1,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = torch.randn(2, 3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "alpha": 1.0,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0], dtype=np.float64)
    input_dict = {
        "alpha": 0.0,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {
        "alpha": 1.0,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    base = np.linspace(-2.0, 2.0, 21, dtype=np.float32)
    input_arr = base[::3]
    input_dict = {
        "alpha": 0.8,
        "inplace": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.tensor([[-3.0], [-1.0], [0.0], [2.0]], dtype=torch.float32).numpy()
    input_dict = {
        "alpha": -1.0,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input_arr = np.empty((0, 5), dtype=np.float32)
    input_dict = {
        "alpha": 1.5,
        "inplace": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ELU"] = elu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ELU'.")


check_valid('torch.nn.ELU', generated_inputs['torch.nn.ELU'], lib="torch", suffix=0)
