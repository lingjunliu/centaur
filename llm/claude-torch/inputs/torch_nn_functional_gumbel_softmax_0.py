
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def gumbel_softmax_inputs():
    list_of_inputs = []
    
    logits = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tau = 1.0
    hard = False
    eps = 1e-10
    dim = -1
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "eps": eps,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = torch.randn(4, 5).numpy()
    tau = 0.5
    hard = True
    eps = 1e-8
    dim = 1
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "eps": eps,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = torch.randn(2, 3, 4).numpy()
    tau = 2.0
    hard = False
    eps = 1e-12
    dim = 2
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "eps": eps,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    tau = 0.1
    hard = True
    eps = 1e-10
    dim = 0
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "eps": eps,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    tau = 10.0
    hard = False
    eps = 1e-9
    dim = -1
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "eps": eps,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = torch.randn(3, 7).numpy()
    tau = 0.01
    hard = True
    eps = 1e-11
    dim = 1
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "eps": eps,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = torch.randn(2, 3, 4, 5).numpy()
    tau = 1.5
    hard = False
    eps = 1e-10
    dim = 3
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "eps": eps,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = torch.tensor([[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]]).numpy()
    tau = 0.75
    hard = False
    eps = 1e-10
    dim = 1
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "eps": eps,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits =

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.gumbel_softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.gumbel_softmax'.")


check_valid('torch.nn.functional.gumbel_softmax', generated_inputs['torch.nn.functional.gumbel_softmax'], lib="torch", suffix=0)
