
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def selu_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32).numpy()
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.full((2, 3), -2.5, dtype=torch.float32).numpy()
    inplace = True
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[1.5, -0.5, 3.0], [-2.0, 0.0, 2.0]], dtype=torch.float64).numpy()
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(24, dtype=torch.float32).view(2, 3, 4).numpy()
    inplace = True
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros((1, 3, 4, 4), dtype=torch.float32).numpy()
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-1.5, 0.0, 2.5]], dtype=torch.float16).numpy()
    inplace = True
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor(3.14, dtype=torch.float32).numpy()
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([], dtype=torch.float32).numpy()
    inplace = True
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(12., dtype=torch.float32).view(3, 4).t().numpy()
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(10, dtype=torch.float32)[::2].numpy()
    inplace = True
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([1e-6, -1e-6, 20.0, -20.0], dtype=torch.float64).numpy()
    inplace = False
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.empty((0, 3), dtype=torch.float32).numpy()
    inplace = True
    input_dict = {"input": input, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.selu"] = selu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.selu'.")


check_valid('torch.nn.functional.selu', generated_inputs['torch.nn.functional.selu'], lib="torch", suffix=0)
