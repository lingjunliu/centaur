
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def gumbel_softmax_inputs():
    list_of_inputs = []

    # Input 1
    logits = np.array([1.0, 2.0, 3.0])
    tau = 1.0
    hard = False
    dim = -1
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    logits = np.array([[1.0, 2.0], [3.0, 4.0]])
    tau = 0.5
    hard = True
    dim = 1
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    logits = np.random.rand(2, 3, 4)
    tau = 0.75
    hard = False
    dim = 2
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    logits = np.array([-1.0, -2.0, -3.0])
    tau = 0.2
    hard = True
    dim = 0
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    logits = np.array([[0.1, 0.2], [0.3, 0.4]])
    tau = 1.5
    hard = False
    dim = -1
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    logits = np.random.rand(5, 2)
    tau = 0.9
    hard = True
    dim = 1
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    logits = np.array([[-1.0, 2.0], [3.0, -4.0]])
    tau = 0.6
    hard = False
    dim = 0
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    logits = np.random.rand(1, 10)
    tau = 1.1
    hard = True
    dim = 1
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    logits = np.array([1.0, 1.0, 1.0])
    tau = 0.8
    hard = False
    dim = -1
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    logits = np.array([[-0.5, 0.5]])
    tau = 1.2
    hard = True
    dim = 1
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    logits = np.random.rand(4,5,6,7)
    tau = 0.3
    hard = False
    dim = 2
    input_dict = {"logits": logits, "tau": tau, "hard": hard, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.gumbel_softmax_2"] = gumbel_softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.gumbel_softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.gumbel_softmax_2'.")

check_valid('torch.nn.functional.gumbel_softmax', generated_inputs['torch.nn.functional.gumbel_softmax_2'], lib="torch", suffix=2)
