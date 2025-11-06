
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def erfcx_inputs():
    list_of_inputs = []

    inp = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    inp = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    inp = torch.tensor([[-1e-6, 0.0, 1e-6],
                        [1e-3, -1e-3, 2e-3]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    inp = torch.tensor([[[-20.0, -5.0],
                         [0.0, 5.0]],
                        [[10.0, 20.0],
                         [-10.0, -15.0]]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    inp = torch.tensor([], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    inp = torch.empty((2, 0, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    base = torch.arange(24., dtype=torch.float64).reshape(2, 3, 4)
    inp = base.transpose(0, 2).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    base = torch.linspace(-5.0, 5.0, steps=20, dtype=torch.float32).reshape(4, 5)
    inp = base[:, ::2].numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    inp = torch.tensor([1e4, -1e4, 50.0, -50.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    inp = torch.tensor(-3.5, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    inp = torch.tensor([float('inf'), float('-inf'), float('nan')], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": inp}))

    return list_of_inputs

generated_inputs["torch.special.erfcx"] = erfcx_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.erfcx' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.erfcx'.")


check_valid('torch.special.erfcx', generated_inputs['torch.special.erfcx'], lib="torch", suffix=0)
