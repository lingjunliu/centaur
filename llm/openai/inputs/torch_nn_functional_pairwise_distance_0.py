
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def pairwise_distance_inputs():
    list_of_inputs = []

    x1 = torch.tensor([[0.0, 1.0, 2.0],
                       [3.0, 4.0, 5.0]], dtype=torch.float32).numpy()
    x2 = torch.tensor([[1.0, 1.0, 1.0],
                       [1.0, 2.0, 3.0]], dtype=torch.float32).numpy()
    p = 2.0
    eps = 1e-6
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.tensor([[-1.0, -2.0, -3.0, -4.0],
                       [5.5, -6.5, 7.5, -8.5],
                       [9.0, 0.0, -1.0, 2.0]], dtype=torch.float64).numpy()
    x2 = torch.tensor([[4.0, 3.0, 2.0, 1.0],
                       [-5.5, 6.5, -7.5, 8.5],
                       [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64).numpy()
    p = 1.0
    eps = 0.0
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0],
                       [6.0, -7.0, 8.0, -9.0, 10.0],
                       [0.5, -0.5, 0.5, -0.5, 0.5]], dtype=torch.float32).numpy()
    x2 = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0]], dtype=torch.float32).numpy()
    p = 2.5
    eps = 1e-12
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.tensor([[1e5, -2e5, 3e5, -4e5],
                       [5e5, -6e5, 7e5, -8e5]], dtype=torch.float32).numpy()
    x2 = torch.tensor([[-1e5, 2e5, -3e5, 4e5],
                       [-5e5, 6e5, -7e5, 8e5]], dtype=torch.float32).numpy()
    p = 3.0
    eps = 1e-4
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float16).numpy()
    x2 = torch.tensor([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0]], dtype=torch.float16).numpy()
    p = float("inf")
    eps = 1e-6
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.tensor([[[1.0, -1.0, 2.0, -2.0],
                        [3.0, -3.0, 4.0, -4.0],
                        [5.0, -5.0, 6.0, -6.0]],
                       [[-1.0, 1.0, -2.0, 2.0],
                        [-3.0, 3.0, -4.0, 4.0],
                        [-5.0, 5.0, -6.0, 6.0]]], dtype=torch.float32).numpy()
    x2 = torch.tensor([[[0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5]],
                       [[-0.5, -0.5, -0.5, -0.5],
                        [-0.5, -0.5, -0.5, -0.5],
                        [-0.5, -0.5, -0.5, -0.5]]], dtype=torch.float32).numpy()
    p = 2.0
    eps = 1e-9
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.arange(2*5*3*2, dtype=torch.float32).reshape(2, 5, 3, 2).numpy()
    x2 = (torch.ones((1, 5, 3, 2), dtype=torch.float32) * 0.5).numpy()
    p = 1.5
    eps = 1e-8
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    x1 = base.transpose(1, 2).contiguous().numpy()  # shape (2, 4, 3)
    x2 = torch.zeros((2, 1, 3), dtype=torch.float32).numpy()
    p = 2.0
    eps = 1e-6
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.tensor([[1e-8, -1e-8],
                       [1e-12, -1e-12]], dtype=torch.float64).numpy()
    x2 = torch.tensor([[0.0, 0.0],
                       [0.0, 0.0]], dtype=torch.float64).numpy()
    p = 2.0
    eps = 1e-20
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=torch.float32).numpy()
    x2 = torch.tensor([[1.0],
                       [4.0]], dtype=torch.float32).numpy()
    p = 1.0
    eps = 1e-12
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.tensor([[[1.0], [2.0]],
                       [[-3.0], [4.0]],
                       [[5.5], [-6.5]]], dtype=torch.float32).numpy()  # shape (3,2,1)
    x2 = torch.tensor([[[0.5], [-0.5]],
                       [[0.5], [-0.5]],
                       [[0.5], [-0.5]]], dtype=torch.float32).numpy()
    p = 3.5
    eps = 1e-7
    keepdim = True
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    x1 = torch.linspace(-3.0, 3.0, steps=28, dtype=torch.float32).reshape(4, 7).numpy()
    x2 = torch.zeros((4, 7), dtype=torch.float32).numpy()
    p = 100.0
    eps = 1.0
    keepdim = False
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "p": p, "eps": eps, "keepdim": keepdim}))

    return list_of_inputs

generated_inputs["torch.nn.functional.pairwise_distance"] = pairwise_distance_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pairwise_distance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pairwise_distance'.")


check_valid('torch.nn.functional.pairwise_distance', generated_inputs['torch.nn.functional.pairwise_distance'], lib="torch", suffix=0)
