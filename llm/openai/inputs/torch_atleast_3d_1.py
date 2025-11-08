
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def atleast_3d_inputs():
    list_of_inputs = []

    input = torch.tensor(5, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-3.14, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1.0, -2.0, 0.0, 3.5, -4.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([1+2j, -3-4j, 0+0j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.arange(9, dtype=torch.int64).view(3, 3).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([[1.5, -2.25], [3.0, 4.5], [-5.5, 0.0]], dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = (torch.arange(24).view(2, 3, 4) > 10).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty(2, 0, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.randn(2, 1, 3, 1, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.empty(0, dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    t = torch.arange(12, dtype=torch.float64).view(3, 4)
    input = torch.flip(t, dims=[1]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.atleast_3d_1"] = atleast_3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.atleast_3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atleast_3d_1'.")


check_valid('torch.atleast_3d', generated_inputs['torch.atleast_3d_1'], lib="torch", suffix=1)
