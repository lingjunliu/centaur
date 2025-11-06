
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def moveaxis_inputs_2():
    list_of_inputs = []

    inp = torch.arange(5., dtype=torch.float32).numpy()
    input_dict = {"input": inp, "source": (0,), "destination": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.arange(6, dtype=torch.int64).reshape(2, 3).numpy()
    input_dict = {"input": inp, "source": (0,), "destination": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 4, 5, dtype=torch.float16).numpy()
    input_dict = {"input": inp, "source": (0,), "destination": (-1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.tensor([[[True], [False]], [[False], [True]], [[True], [True]]], dtype=torch.bool).numpy()
    input_dict = {"input": inp, "source": (1, 2), "destination": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 4, 5, dtype=torch.float64).numpy()
    input_dict = {"input": inp, "source": (-1, -3), "destination": (0, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.ones(1, 1, 1, dtype=torch.float32).numpy()
    input_dict = {"input": inp, "source": (0,), "destination": (2,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.arange(1*2*3*4*5, dtype=torch.int32).reshape(1, 2, 3, 4, 5).numpy()
    input_dict = {"input": inp, "source": (0, 2, 4), "destination": (4, 0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    c64 = (torch.randn(1, 1) + 1j * torch.randn(1, 1)).to(torch.complex64).numpy()
    input_dict = {"input": c64, "source": (-2, -1), "destination": (1, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randint(0, 256, (2, 1, 3, 1, 4, 1), dtype=torch.uint8).numpy()
    input_dict = {"input": inp, "source": (1, 3, 5), "destination": (5, 3, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 4, dtype=torch.float32).numpy()
    input_dict = {"input": inp, "source": (0, 1, 2), "destination": (2, 0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = (torch.rand(3, 1, 1, 2) > 0.5).numpy()
    input_dict = {"input": inp, "source": (2, 0), "destination": (0, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = torch.randn(4, 5, 6, dtype=torch.float64)
    imag = torch.randn(4, 5, 6, dtype=torch.float64)
    c128 = (real + 1j * imag).to(torch.complex128).numpy()
    input_dict = {"input": c128, "source": (-3, -2), "destination": (-1, -2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.moveaxis_2"] = moveaxis_inputs_2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.moveaxis_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.moveaxis_2'.")


check_valid('torch.moveaxis', generated_inputs['torch.moveaxis_2'], lib="torch", suffix=2)
