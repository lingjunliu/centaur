
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def relu_inputs():
    list_of_inputs = []

    input = torch.tensor(-1.5, dtype=torch.float32).numpy()
    input_dict = {"inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([-3.0, 0.0, 2.5, -0.0, 7.2], dtype=torch.float32).numpy()
    input_dict = {"inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 3, dtype=torch.float64).numpy()
    input_dict = {"inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-5, 5, steps=24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    input_dict = {"inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = torch.empty((1, 3, 64, 64), dtype=torch.float32).uniform_(-1.0, 1.0)
    input = t.numpy()
    input_dict = {"inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 1.0], dtype=np.float64)
    input_dict = {"inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.empty(0, dtype=torch.float32).numpy()
    input_dict = {"inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.empty((2, 0), dtype=np.float32)
    input_dict = {"inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.linspace(-1000, 1000, num=25, dtype=np.float16).reshape(5, 5)
    input_dict = {"inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.arange(24, dtype=np.float32).reshape(4, 6).T
    input_dict = {"inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.array([-1e-7, 0.0, 1e-7, -1e20, 1e20, -3.4e38, 3.4e38], dtype=np.float32)
    input_dict = {"inplace": False, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(1, 1, 1, 2, 3, 1).astype(np.float32)
    input_dict = {"inplace": True, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ReLU"] = relu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReLU'.")


check_valid('torch.nn.ReLU', generated_inputs['torch.nn.ReLU'], lib="torch", suffix=0)
