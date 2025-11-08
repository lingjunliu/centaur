
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def poisson_nll_loss_inputs():
    list_of_inputs = []
    
    # Input 1
    input_arr = torch.tensor([0.0, 1.0, -1.0, 2.5], dtype=torch.float32).numpy()
    target = torch.tensor([0.0, 1.0, 3.0, 2.0], dtype=torch.float32).numpy()
    log_input = True
    full = False
    eps = 1e-8
    reduction = 'mean'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.tensor([[0.0, 0.5, 10.0],
                              [3.0, 1.2, 0.0]], dtype=torch.float64).numpy()
    target = torch.tensor([[0.0, 1.0, 7.0],
                           [2.0, 0.0, 3.0]], dtype=torch.float64).numpy()
    log_input = False
    full = False
    eps = 1e-12
    reduction = 'sum'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.tensor([[[-0.5, 0.0],
                               [1.2, -2.0]],
                              [[0.3, 0.7],
                               [-1.1, 2.5]]], dtype=torch.float32).numpy()
    target = torch.tensor([[[0.0, 2.0],
                            [1.0, 3.0]],
                           [[4.0, 0.0],
                            [2.0, 1.0]]], dtype=torch.float32).numpy()
    log_input = True
    full = True
    eps = 1e-8
    reduction = 'none'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.tensor([0.0, 0.2, 1.5, 3.0, 5.0], dtype=torch.float32).numpy()
    target = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0], dtype=torch.float32).numpy()
    log_input = False
    full = False
    eps = 1e-8
    reduction = 'none'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.tensor([[-5.0],
                              [0.0],
                              [3.0]], dtype=torch.float64).numpy()
    target = torch.tensor([[0.0],
                           [10.0],
                           [2.0]], dtype=torch.float64).numpy()
    log_input = True
    full = True
    eps = 1e-10
    reduction = 'mean'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = (torch.tensor([[[[0.1, 0.2],
                                 [0.3, 0.4],
                                 [0.5, 0.6]]],
                               [[[1.1, 1.2],
                                 [1.3, 1.4],
                                 [1.5, 1.6]]]], dtype=torch.float64).numpy())
    target = (torch.tensor([[[[0.0, 1.0],
                              [2.0, 0.0],
                              [1.0, 3.0]]],
                            [[[2.0, 0.0],
                              [1.0, 4.0],
                              [0.0, 2.0]]]], dtype=torch.float64).numpy())
    log_input = False
    full = False
    eps = 0.0
    reduction = 'sum'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = np.array(0.7, dtype=np.float32)
    target = np.array(2.0, dtype=np.float32)
    log_input = False
    full = True
    eps = 1e-6
    reduction = 'mean'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.tensor([[[-2.0],
                               [-0.1],
                               [0.0],
                               [2.2]]], dtype=torch.float64).numpy()
    target = torch.tensor([[[0.0],
                            [1.0],
                            [2.0],
                            [3.0]]], dtype=torch.float64).numpy()
    log_input = True
    full = False
    eps = 1e-8
    reduction = 'none'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.tensor([[20.0, 50.0],
                              [5.0, 100.0]], dtype=torch.float32).numpy()
    target = torch.tensor([[15.0, 0.0],
                           [40.0, 5.0]], dtype=torch.float32).numpy()
    log_input = False
    full = True
    eps = 1e-4
    reduction = 'mean'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.tensor([-3.0, -0.5, 0.1, 1.0, 2.5], dtype=torch.float64).numpy()
    target = torch.tensor([0.5, 1.5, 2.0, 0.0, 3.5], dtype=torch.float64).numpy()
    log_input = True
    full = False
    eps = 1e-7
    reduction = 'sum'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.tensor([[[1e-8],
                               [1e-4],
                               [1e-2]],
                              [[0.3],
                               [5.0],
                               [10.0]]], dtype=torch.float64).numpy()
    target = torch.tensor([[[0.0],
                            [0.0],
                            [1.0]],
                           [[2.0],
                            [3.0],
                            [4.0]]], dtype=torch.float64).numpy()
    log_input = False
    full = False
    eps = 1e-9
    reduction = 'none'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.tensor([-20.0, -0.0001], dtype=torch.float32).numpy()
    target = torch.tensor([0.0, 1.0], dtype=torch.float32).numpy()
    log_input = True
    full = True
    eps = 1e-8
    reduction = 'sum'
    input_dict = {
        "input": input_arr,
        "target": target,
        "log_input": log_input,
        "full": full,
        "eps": eps,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.poisson_nll_loss"] = poisson_nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.poisson_nll_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.poisson_nll_loss'.")


check_valid('torch.nn.functional.poisson_nll_loss', generated_inputs['torch.nn.functional.poisson_nll_loss'], lib="torch", suffix=0)
