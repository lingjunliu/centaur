
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def movedim_inputs():
    list_of_inputs = []
    
    input_arr = torch.arange(5).numpy()
    source = 0
    destination = 0
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3).numpy()
    source = 0
    destination = 1
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones((2, 3), dtype=torch.float64).numpy()
    source = 1
    destination = 0
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 4, 5).numpy()
    source = 2
    destination = 0
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 4, 5).numpy()
    source = -1
    destination = 1
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 3, 4).numpy()
    source = 1
    destination = -1
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.zeros((0, 3, 2), dtype=torch.float32).numpy()
    source = 0
    destination = 2
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(0, 255, (2, 2, 2), dtype=torch.uint8).numpy()
    source = -3
    destination = -1
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 3, 4, 5, dtype=torch.float16).numpy()
    source = 4
    destination = 0
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 1, 4).numpy()
    source = 2
    destination = 1
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[True, False], [False, True], [True, True]], dtype=torch.bool).unsqueeze(0).numpy()
    source = -2
    destination = -3
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.complex(torch.randn(2, 3, 4), torch.randn(2, 3, 4)).numpy()
    source = 0
    destination = 2
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 4, 5).numpy()
    source = 3
    destination = 3
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(7).unsqueeze(0).unsqueeze(-1).numpy()
    source = -3
    destination = -1
    input_dict = {
        "input": input_arr,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.movedim_1"] = movedim_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.movedim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.movedim_1'.")


check_valid('torch.movedim', generated_inputs['torch.movedim_1'], lib="torch", suffix=1)
