
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def permute_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 5).numpy()
    dims = (2, 0, 1)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = torch.ones((2, 3)).numpy()
    dims = (1, 0)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = torch.randn(2, 3, 4, 5).numpy()
    dims = (0, 2, 3, 1)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = torch.arange(5).numpy()
    dims = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = torch.arange(1 * 2 * 3 * 4 * 5, dtype=torch.float32).view(1, 2, 3, 4, 5).numpy()
    dims = (4, 0, 2, 3, 1)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = torch.arange(24, dtype=torch.int32).view(2, 3, 4).numpy()
    dims = (1, 2, 0)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = (torch.rand(2, 2, 3) > 0.5).numpy()
    dims = (2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = torch.randn(3, 4, 5).numpy()
    dims = (-1, -3, -2)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = torch.empty((0, 3, 4), dtype=torch.float64).numpy()
    dims = (2, 0, 1)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    base = torch.arange(3 * 4 * 5).view(3, 4, 5).numpy()
    input = base[:, ::-1, ::2]
    dims = (2, 0, 1)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = np.asfortranarray(np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4))
    dims = (1, 0, 2)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = (torch.randn(4, 5) + 1j * torch.randn(4, 5)).numpy()
    dims = (1, 0)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
    dims = (1, 2, 0)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    input = torch.arange(1 * 1 * 2 * 3 * 1 * 4, dtype=torch.float16).view(1, 1, 2, 3, 1, 4).numpy()
    dims = (5, 2, 3, 1, 4, 0)
    list_of_inputs.append(copy.deepcopy({"input": input, "dims": dims}))

    return list_of_inputs

generated_inputs["torch.permute"] = permute_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.permute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.permute'.")


check_valid('torch.permute', generated_inputs['torch.permute'], lib="torch", suffix=0)
