
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def atleast_3d_inputs():
    list_of_inputs = []

    a1 = torch.tensor(7.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a1}))

    a2 = torch.tensor([-3, 0, 5], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a2}))

    a3 = torch.tensor([[1.2, -2.3], [3.4, 4.5]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a3}))

    a4 = torch.arange(-24, 0, dtype=torch.int32).view(2, 3, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a4}))

    a5 = torch.randn(2, 3, 4, 5, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a5}))

    a6 = torch.tensor([1 + 2j, -3 - 4j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a6}))

    a7 = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a7}))

    a8 = torch.empty(2, 0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a8}))

    a9 = torch.arange(24, dtype=torch.float32).view(2, 3, 4).transpose(1, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a9}))

    a10 = torch.tensor([[True, False], [False, True]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a10}))

    a11 = torch.tensor(-5, dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a11}))

    a12 = torch.arange(-128, 128, dtype=torch.int8).view(16, 16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": a12}))

    return list_of_inputs

generated_inputs["torch.atleast_3d_2"] = atleast_3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.atleast_3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atleast_3d_2'.")


check_valid('torch.atleast_3d', generated_inputs['torch.atleast_3d_2'], lib="torch", suffix=2)
