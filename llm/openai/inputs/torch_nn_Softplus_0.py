
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def softplus_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor(0.0, dtype=torch.float32).numpy()
    input_dict = {"beta": 1.0, "threshold": 20.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.tensor([-1.0, 0.0, 1.0, 10.0, -10.0], dtype=torch.float32).numpy()
    input_dict = {"beta": 1.0, "threshold": 20.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(2, 3, dtype=torch.float64).numpy()
    input_dict = {"beta": 2.0, "threshold": 20.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = (-torch.rand(2, 2, 2, dtype=torch.float32) * 5.0).numpy()
    input_dict = {"beta": 0.5, "threshold": 10.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(1, 3, 4, 4, dtype=torch.float16).numpy()
    input_dict = {"beta": 1.0, "threshold": 15.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.tensor([1000.0, -1000.0, 21.0, -21.0, 0.1], dtype=torch.float32).numpy()
    input_dict = {"beta": 1.0, "threshold": 20.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.tensor([-1e-8, 1e-8, -1e-4, 1e-4], dtype=torch.float64).numpy()
    input_dict = {"beta": 10.0, "threshold": 20.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    input_dict = {"beta": 1.5, "threshold": 30.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = torch.randn(5, dtype=torch.float32).numpy()
    input_dict = {"beta": 100.0, "threshold": 50.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.linspace(-3, 3, steps=12, dtype=torch.float32).reshape(2, 2, 3).numpy()
    input_dict = {"beta": 0.01, "threshold": 1.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = torch.randn(10, 10, dtype=torch.float32).numpy()
    input_dict = {"beta": 1.0, "threshold": 5.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_tensor = torch.tensor([], dtype=torch.float32).numpy()
    input_dict = {"beta": 1.0, "threshold": 20.0, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.Softplus"] = softplus_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softplus'.")


check_valid('torch.nn.Softplus', generated_inputs['torch.nn.Softplus'], lib="torch", suffix=0)
