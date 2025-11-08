
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def zeta_inputs():
    list_of_inputs = []

    x = torch.tensor([2.0, 3.5, 5.0], dtype=torch.float32).numpy()
    q = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor(2.3, dtype=torch.float64).numpy()
    q = torch.tensor(1.0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([[2.5], [4.0]], dtype=torch.float64).numpy()
    q = torch.tensor([0.25, 1.5, 3.0], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([[[2.0, 2.5], [3.0, 3.5]],
                      [[4.0, 4.5], [5.0, 5.5]]], dtype=torch.float32).numpy()
    q = torch.tensor([[[0.5, 1.2], [0.75, 2.0]],
                      [[1.5, 2.5], [3.0, 0.8]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.linspace(1.5, 5.0, steps=6, dtype=torch.float32).reshape(2, 3).numpy()
    q = torch.full((2, 3), 1.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([2.0, 2.1, 10.0], dtype=torch.float64).numpy()
    q = torch.tensor(1.25, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([[2.2, 2.4, 3.1],
                      [4.0, 6.0, 8.0]], dtype=torch.float64).numpy()
    q = torch.tensor([[0.5],
                      [1.5]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = (torch.arange(12, dtype=torch.float32).reshape(3, 4) / 4.0 + 2.0).numpy()
    q = torch.linspace(0.25, 2.0, steps=4, dtype=torch.float32).reshape(1, 4).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([[[[2.2, 3.3, 4.4]],
                        [[5.5, 6.6, 7.7]]]], dtype=torch.float64).numpy()
    q = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.full((4,), 2.2, dtype=torch.float32).numpy()
    q = torch.tensor([0.3, 0.7, 1.1, 2.3], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([[2.01, 2.5],
                      [3.2, 4.8],
                      [6.0, 9.0]], dtype=torch.float64).numpy()
    q = torch.tensor([1.0, 0.9], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    return list_of_inputs

generated_inputs["torch.special.zeta_1"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.zeta_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.zeta_1'.")


check_valid('torch.special.zeta', generated_inputs['torch.special.zeta_1'], lib="torch", suffix=1)
