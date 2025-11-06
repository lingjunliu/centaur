
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def feature_alpha_dropout_inputs():
    list_of_inputs = []

    # 1
    input = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).numpy()
    p = 0.0
    training = True
    inplace = False
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input = torch.tensor([[-1.0, 0.0, 1.0], [2.5, -3.5, 4.0]], dtype=torch.float64).numpy()
    p = 0.05
    training = True
    inplace = True
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    p = 0.2
    training = True
    inplace = False
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input = torch.randn(1, 3, 8, 8, dtype=torch.float32).numpy()
    p = 0.5
    training = True
    inplace = False
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input = torch.linspace(-5, 5, steps=64, dtype=torch.float32).reshape(2, 2, 2, 2, 2, 2).numpy()
    p = 0.9
    training = True
    inplace = False
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input = torch.ones(4, 4, dtype=torch.float32).mul_(10).numpy()
    p = 0.75
    training = False
    inplace = True
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input = torch.tensor([[0.001, -0.002, 0.003, -0.004, 0.005]], dtype=torch.float64).numpy()
    p = 0.3
    training = False
    inplace = False
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input = torch.empty(0, 10, dtype=torch.float32).numpy()
    p = 0.1
    training = True
    inplace = False
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input = torch.tensor([[-1000.0, 0.0], [1000.0, -1e6]], dtype=torch.float32).numpy()
    p = 0.4
    training = True
    inplace = True
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input = torch.randn(3, 4, 5, dtype=torch.float64).numpy()
    p = 0.95
    training = True
    inplace = False
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input = (torch.randn(2, 3, dtype=torch.float32) * 1e-3).numpy()
    p = 0.01
    training = True
    inplace = True
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input = torch.linspace(-2.0, 2.0, steps=10, dtype=torch.float32).reshape(1, 10).numpy()
    p = 0.66
    training = False
    inplace = False
    input_dict = {"input": input, "p": p, "training": training, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.feature_alpha_dropout"] = feature_alpha_dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.feature_alpha_dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.feature_alpha_dropout'.")


check_valid('torch.nn.functional.feature_alpha_dropout', generated_inputs['torch.nn.functional.feature_alpha_dropout'], lib="torch", suffix=0)
