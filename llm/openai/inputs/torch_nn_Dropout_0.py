
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def dropout_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    p = 0.0
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(20, 16, dtype=torch.float32).numpy()
    p = 0.5
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    p = 0.2
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.ones((0, 5), dtype=torch.float32).numpy()
    p = 0.8
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.rand(4, 4, dtype=torch.float64).numpy()
    p = 1.0
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 3, 32, 32, dtype=torch.float16).numpy()
    p = 0.3
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor(2.0, dtype=torch.float32).numpy()
    p = 0.05
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-1, 1, steps=10, dtype=torch.float32).numpy()
    p = 0.7
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros(5, dtype=torch.float32).numpy()
    p = 0.4
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 4, 6, 8, 10, dtype=torch.float32).numpy()
    p = 0.9
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-1000.0, 0.0, 2000.0]], dtype=torch.float32).numpy()
    p = 0.15
    inplace = False
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 224, 224, dtype=torch.float32).numpy()
    p = 0.6
    inplace = True
    input_dict = {"p": p, "inplace": inplace, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.Dropout"] = dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Dropout'.")


check_valid('torch.nn.Dropout', generated_inputs['torch.nn.Dropout'], lib="torch", suffix=0)
