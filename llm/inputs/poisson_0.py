
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def poisson_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, small rates
    rates = torch.rand(5).numpy() * 2
    input_dict = {"input": rates, "generator": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, larger rates
    rates = torch.rand(2, 3).numpy() * 10
    input_dict = {"input": rates, "generator": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, mix of small and large rates
    rates = torch.rand(2, 2, 2).numpy() * 5
    input_dict = {"input": rates, "generator": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar tensor
    rates = torch.tensor(3.5).numpy()
    input_dict = {"input": rates, "generator": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with some zero rates
    rates = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0]).numpy()
    input_dict = {"input": rates, "generator": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.poisson"] = poisson_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.poisson' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.poisson'.")

check_valid('torch.poisson', generated_inputs['torch.poisson'], lib="torch", suffix=0)
