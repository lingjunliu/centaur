
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def movedim_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(5, dtype=torch.float32).numpy()
    source = (0,)
    destination = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 2
    input_arr = torch.arange(6, dtype=torch.float32).reshape(2, 3).numpy()
    source = (0,)
    destination = (1,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 3
    input_arr = torch.arange(8, dtype=torch.float32).reshape(4, 2).numpy()
    source = (1,)
    destination = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 4
    input_arr = torch.randn(3, 2, 1, dtype=torch.float32).numpy()
    source = (1, 2)
    destination = (-3, -2)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 5
    input_arr = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    source = (0,)
    destination = (-1,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 6
    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    source = (0, 2)
    destination = (2, 0)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 7
    input_arr = torch.arange(120).reshape(2, 3, 4, 5).numpy()
    source = (-1, -3)
    destination = (0, 3)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 8
    input_arr = torch.ones(1, 2, 3, 4, 5, dtype=torch.float16).numpy()
    source = (0, 1, 2)
    destination = (2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 9
    input_arr = torch.randn(2, 1, 3, 1, 4, 1, dtype=torch.float32).numpy()
    source = (1, 3, 5)
    destination = (5, 3, 1)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 10
    input_arr = torch.randn(4, 5, 6, dtype=torch.double).numpy()
    source = (2,)
    destination = (0,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 11
    input_arr = (torch.randn(2, 3, 4, dtype=torch.float32) + 1j * torch.randn(2, 3, 4, dtype=torch.float32)).numpy()
    source = (-2,)
    destination = (-1,)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    # Input 12
    input_arr = torch.randint(0, 100, (2, 2, 2, 2, 2), dtype=torch.int8).numpy()
    source = (4, 0)
    destination = (0, 4)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "source": source, "destination": destination}))

    return list_of_inputs

generated_inputs["torch.movedim_2"] = movedim_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.movedim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.movedim_2'.")


check_valid('torch.movedim', generated_inputs['torch.movedim_2'], lib="torch", suffix=2)
