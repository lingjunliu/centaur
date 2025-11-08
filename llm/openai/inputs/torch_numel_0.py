
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def numel_inputs():
    list_of_inputs = []

    input = torch.tensor(42).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor(True).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([], dtype=torch.float32).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros((3, 4), dtype=torch.float32).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn((2, 3, 4), dtype=torch.float64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.ones((1, 2, 3, 4), dtype=torch.int64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.randn(1, 2, 3, 4, 5, dtype=torch.float32)
    imag = torch.randn(1, 2, 3, 4, 5, dtype=torch.float32)
    input = torch.complex(real, imag).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros((4, 0, 3), dtype=torch.float32).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(12).reshape(3, 4).t().numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn((2, 1, 3, 1, 4, 2), dtype=torch.float16).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randint(0, 256, (3, 3), dtype=torch.uint8).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-1.0, -2.5, 0.0], [3.14, -100.0, 2.71]], dtype=torch.float64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.numel"] = numel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.numel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.numel'.")


check_valid('torch.numel', generated_inputs['torch.numel'], lib="torch", suffix=0)
