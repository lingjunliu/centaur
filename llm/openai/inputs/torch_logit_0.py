
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logit_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64).numpy()
    eps = 0.0
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 2
    input = torch.tensor([[0.2, 0.8, 0.3],
                          [0.01, 0.99, 0.5]], dtype=torch.float32).numpy()
    eps = 1e-6
    out = np.zeros_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 3
    input = torch.rand((2, 3, 4), dtype=torch.float16).numpy()
    eps = 1e-3
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 4
    input = torch.rand((2, 2, 2, 3), dtype=torch.float64).numpy()
    eps = 1e-12
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 5
    input = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32).numpy()
    eps = 1e-2
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 6
    input = torch.tensor([-0.1, 0.5, 1.1], dtype=torch.float32).numpy()
    eps = 0.1
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 7
    input = torch.tensor([], dtype=torch.float32).numpy()
    eps = 0.5
    out = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 8
    input = torch.tensor(0.73, dtype=torch.float64).numpy()
    eps = 0.01
    out = np.empty((), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 9
    base = torch.linspace(0.0, 1.0, steps=20, dtype=torch.float32).reshape(4, 5).numpy()
    input = base[:, ::2]
    eps = 1e-5
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 10
    input = torch.tensor([1e-12, 1.0 - 1e-12, 5e-16, 1.0 - 5e-16], dtype=torch.float64).numpy()
    eps = 1e-15
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 11
    input = torch.empty((2, 0, 3), dtype=torch.float16).numpy()
    eps = 0.2
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    # Input 12
    input = torch.tensor([np.nan, np.inf, -np.inf, 0.0, 1.0, 0.5], dtype=torch.float64).numpy()
    eps = 0.5
    out = np.empty_like(input)
    list_of_inputs.append(copy.deepcopy({"input": input, "eps": eps, "out": out}))

    return list_of_inputs

generated_inputs["torch.logit"] = logit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logit'.")


check_valid('torch.logit', generated_inputs['torch.logit'], lib="torch", suffix=0)
