
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def fftshift_inputs():
    list_of_inputs = []

    input = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(-5, 5, dtype=torch.int32).numpy()
    dim = -1
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.complex64).numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-1.0, 1.0, steps=15, dtype=torch.float64).reshape(3, 5).numpy()
    dim = 1
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).numpy()
    dim = -2
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(30, dtype=torch.int64).reshape(5, 1, 6).numpy()
    dim = 2
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(60, dtype=torch.float64).reshape(2, 2, 3, 5).to(torch.complex128).numpy()
    dim = -4
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([True, False, True, False, True], dtype=torch.bool).numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(7, dtype=torch.float32).reshape(7, 1).numpy()
    dim = 1
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(24, dtype=torch.float16).reshape(1, 2, 1, 3, 4).numpy()
    dim = 3
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-3, -2, -1, 0, 1, 2]], dtype=torch.int16).reshape(1, 6).numpy()
    dim = -1
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 5, 5, dtype=torch.float32).numpy()
    dim = 2
    input_dict = {"input": input, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.fft.fftshift_1"] = fftshift_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.fftshift_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fftshift_1'.")


check_valid('torch.fft.fftshift', generated_inputs['torch.fft.fftshift_1'], lib="torch", suffix=1)
