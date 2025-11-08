
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def poisson_nll_loss_inputs():
    list_of_inputs = []

    # 1: 1D, log_input=True, mean reduction
    inp = torch.tensor([-0.2, 0.0, 0.5, 1.2, -1.5], dtype=torch.float32).numpy()
    tgt = torch.tensor([0.0, 1.0, 2.0, 0.3, 4.0], dtype=torch.float32).numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": True,
        "eps": 1e-8,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2: 2D, log_input=False, sum reduction
    inp = torch.tensor([[0.1, 0.5, 1.0],
                        [2.0, 3.5, 4.2]], dtype=torch.float32).numpy()
    tgt = torch.tensor([[0.0, 1.0, 0.0],
                        [2.0, 1.5, 3.0]], dtype=torch.float32).numpy()
    input_dict = {
        "log_input": False,
        "full": False,
        "size_average": True,
        "eps": 1e-8,
        "reduce": True,
        "reduction": "sum",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3: scalar (0-dim), log_input=False, none reduction
    inp = torch.tensor(2.5, dtype=torch.float32).numpy()
    tgt = torch.tensor(0.0, dtype=torch.float32).numpy()
    input_dict = {
        "log_input": False,
        "full": False,
        "size_average": False,
        "eps": 1e-7,
        "reduce": False,
        "reduction": "none",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4: 3D, log_input=True, full=True
    inp = torch.tensor([[[0.0, -0.5],
                         [1.2, 0.7]],
                        [[-1.0, 0.3],
                         [0.9, -0.2]]], dtype=torch.float32).numpy()
    tgt = torch.tensor([[[0.0, 2.0],
                         [3.0, 0.5]],
                        [[1.0, 0.0],
                         [4.0, 1.0]]], dtype=torch.float32).numpy()
    input_dict = {
        "log_input": True,
        "full": True,
        "size_average": True,
        "eps": 1e-8,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5: 4D, log_input=True, zeros target
    inp = torch.randn(1, 2, 3, 4, dtype=torch.float32).numpy()
    tgt = torch.zeros(1, 2, 3, 4, dtype=torch.float32).numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": True,
        "eps": 1e-8,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6: 2D, float64 dtype, log_input=False, none reduction
    inp = torch.tensor([[0.0, 1.0],
                        [5.0, 10.0]], dtype=torch.float64).numpy()
    tgt = torch.tensor([[0.0, 0.1],
                        [2.0, 3.0]], dtype=torch.float64).numpy()
    input_dict = {
        "log_input": False,
        "full": True,
        "size_average": True,
        "eps": 1e-12,
        "reduce": False,
        "reduction": "none",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7: 2D, log_input=True, sum reduction
    inp = torch.tensor([[0.3, -0.7, 1.1],
                        [0.0, 0.2, -0.2]], dtype=torch.float32).numpy()
    tgt = torch.tensor([[1.0, 0.0, 2.0],
                        [0.5, 0.0, 1.0]], dtype=torch.float32).numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": True,
        "eps": 1e-6,
        "reduce": True,
        "reduction": "sum",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8: 2D, large targets, log_input=True, full=True
    inp = torch.tensor([[2.0, 1.0],
                        [0.5, -0.3]], dtype=torch.float32).numpy()
    tgt = torch.tensor([[50.0, 100.0],
                        [25.0, 40.0]], dtype=torch.float32).numpy()
    input_dict = {
        "log_input": True,
        "full": True,
        "size_average": False,
        "eps": 1e-8,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9: 1D, zeros in input, log_input=False, mean reduction
    inp = torch.tensor([0.0, 0.0, 1.0, 2.0], dtype=torch.float32).numpy()
    tgt = torch.tensor([0.0, 1.0, 0.5, 3.0], dtype=torch.float32).numpy()
    input_dict = {
        "log_input": False,
        "full": False,
        "size_average": True,
        "eps": 1e-6,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10: 3D, small targets, log_input=True, reduction none
    inp = torch.tensor([[[0.1], [-0.1]],
                        [[0.2], [0.0]]], dtype=torch.float32).numpy()
    tgt = torch.tensor([[[0.0], [0.8]],
                        [[1.2], [0.0]]], dtype=torch.float32).numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": False,
        "eps": 1e-8,
        "reduce": False,
        "reduction": "none",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11: 1D, float64, log_input=False, sum reduction
    inp = torch.tensor([0.2, 3.3, 4.4, 5.5], dtype=torch.float64).numpy()
    tgt = torch.tensor([0.0, 1.0, 0.0, 2.5], dtype=torch.float64).numpy()
    input_dict = {
        "log_input": False,
        "full": True,
        "size_average": False,
        "eps": 1e-9,
        "reduce": True,
        "reduction": "sum",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12: 4D, float32, log_input=True, reduction none, reduce False
    inp = torch.randn(2, 1, 2, 2, dtype=torch.float32).numpy()
    tgt = torch.abs(torch.randn(2, 1, 2, 2, dtype=torch.float32)).numpy()
    input_dict = {
        "log_input": True,
        "full": True,
        "size_average": True,
        "eps": 1e-8,
        "reduce": False,
        "reduction": "none",
        "input": inp,
        "target": tgt
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.PoissonNLLLoss"] = poisson_nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.PoissonNLLLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.PoissonNLLLoss'.")


check_valid('torch.nn.PoissonNLLLoss', generated_inputs['torch.nn.PoissonNLLLoss'], lib="torch", suffix=0)
