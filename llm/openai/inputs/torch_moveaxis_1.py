
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def moveaxis_inputs_1():
    list_of_inputs = []

    # Input 1
    input = torch.arange(5, dtype=torch.float32).numpy()
    source = 0
    destination = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 2
    input = torch.randn(2, 3, dtype=torch.float32).numpy()
    source = 0
    destination = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 3
    input = torch.tensor([[1], [2], [3], [4]], dtype=torch.int64).numpy()
    source = 1
    destination = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 4
    input = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    source = -1
    destination = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 5
    input = (torch.rand(1, 5, 1) > 0.5).numpy()
    source = 0
    destination = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 6
    input = torch.randint(0, 256, (3, 4, 5, 6), dtype=torch.uint8).numpy()
    source = 2
    destination = 1
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 7
    input = torch.randn(2, 1, 3, 1, 4, dtype=torch.complex64).numpy()
    source = -3
    destination = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 8
    input = torch.ones(1, 1, 1).numpy()
    source = 1
    destination = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 9
    input = torch.randint(0, 10, (2, 2, 2, 2, 2, 2), dtype=torch.int32).numpy()
    source = 4
    destination = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 10
    input = torch.randn(1, 7, dtype=torch.float64).numpy()
    source = -2
    destination = -1
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 11 (non-contiguous)
    input = torch.arange(60).reshape(3, 4, 5).transpose(0, 1).numpy()
    source = 1
    destination = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    # Input 12
    input = torch.randn(1, 3, 1, 2, dtype=torch.float32).numpy()
    source = 3
    destination = 0
    list_of_inputs.append(copy.deepcopy({"input": input, "source": source, "destination": destination}))

    return list_of_inputs

generated_inputs["torch.moveaxis_1"] = moveaxis_inputs_1()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.moveaxis_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.moveaxis_1'.")


check_valid('torch.moveaxis', generated_inputs['torch.moveaxis_1'], lib="torch", suffix=1)
