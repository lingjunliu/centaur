
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def special_zeta_inputs_2():
    list_of_inputs = []

    x = torch.tensor([2.0, 3.5, 10.0], dtype=torch.float64).numpy()
    q = 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([[-2.0, 0.5], [2.5, 5.0]], dtype=torch.float32).numpy()
    q = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.linspace(0.2, 4.8, steps=8, dtype=torch.float64).reshape(2, 2, 2).numpy()
    q = 0.75
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor(0.25, dtype=torch.float32).numpy()
    q = 3.0
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([], dtype=torch.float64).numpy()
    q = 1.2
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([-10.5, -1.2, 2.2, 20.0], dtype=torch.float64).numpy()
    q = 3.0
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).reshape(9, 1).numpy()
    q = 0.1
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.full((1, 3, 1), 2.2, dtype=torch.float64).numpy()
    q = 5.5
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = (torch.arange(1, 7, dtype=torch.float32) + 0.1).reshape(1, 2, 3, 1).numpy()
    q = 2.7
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([50.0, 100.0, -50.0, -100.0], dtype=torch.float64).numpy()
    q = 0.9
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([0.999999, 1.000001, 2.0], dtype=torch.float64).numpy()
    q = 1.3
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.empty((3, 0, 2), dtype=torch.float64).numpy()
    q = 2.2
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.tensor([-0.75, -2.5, -3.3], dtype=torch.float32).numpy()
    q = 0.3
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    x = torch.linspace(0.2, 2.2, steps=10, dtype=torch.float32).reshape(2, 5).numpy()
    q = 4.0
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    return list_of_inputs

generated_inputs["torch.special.zeta_2"] = special_zeta_inputs_2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.zeta_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.zeta_2'.")


check_valid('torch.special.zeta', generated_inputs['torch.special.zeta_2'], lib="torch", suffix=2)
