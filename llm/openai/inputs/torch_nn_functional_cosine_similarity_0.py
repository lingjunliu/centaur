
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def cosine_similarity_inputs():
    list_of_inputs = []

    # Input 1
    x1 = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32).numpy()
    x2 = torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float32).numpy()
    dim = 0
    eps = 1e-8
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 2
    x1 = torch.tensor([[1.0, 0.0, -1.0],
                       [2.0, -2.0, 3.0]], dtype=torch.float64).numpy()
    x2 = torch.tensor([[-1.0, 1.0, 0.0],
                       [2.0, 1.0, -3.0]], dtype=torch.float64).numpy()
    dim = 1
    eps = 1e-8
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 3
    base = torch.arange(20, dtype=torch.float32).reshape(4, 5) - 5.0
    x1 = base.numpy()
    x2 = torch.flip(base, dims=[0]).numpy()
    dim = 0
    eps = 0.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 4
    x1 = (torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) - 12.0).numpy()
    x2 = (torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) - 11.5).numpy()
    dim = 2
    eps = 1e-12
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 5
    x1 = torch.linspace(-1.0, 1.0, steps=2*3*4, dtype=torch.float32).reshape(2, 3, 4).numpy()
    x2 = (-torch.linspace(-1.0, 1.0, steps=2*3*4, dtype=torch.float32).reshape(2, 3, 4)).numpy()
    dim = -2
    eps = 1e-5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 6
    x1 = (torch.ones((2, 3, 4, 5), dtype=torch.float32) * 2.0).numpy()
    x2 = (torch.arange(2*3*4*5, dtype=torch.float32).reshape(2, 3, 4, 5) * 0.1).numpy()
    dim = -1
    eps = 1e-6
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 7
    x1 = torch.arange(12, dtype=torch.float16).reshape(1, 4, 1, 3).numpy()
    x2 = (torch.arange(12, dtype=torch.float16).reshape(1, 4, 1, 3) + 1).numpy()
    dim = 3
    eps = 1e-3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 8
    x1 = torch.zeros(10, dtype=torch.float32).numpy()
    x2 = torch.ones(10, dtype=torch.float32).numpy()
    dim = 0
    eps = 1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 9
    base = torch.arange(12.0, dtype=torch.float32).reshape(3, 4)
    x1 = base.t().contiguous().numpy()
    x2 = (base + 2.0).t().contiguous().numpy()
    dim = 1
    eps = 1e-8
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 10
    x1 = torch.arange(2*1*3*1*4, dtype=torch.float32).reshape(2, 1, 3, 1, 4).numpy()
    x2 = torch.flip(torch.arange(2*1*3*1*4, dtype=torch.float32).reshape(2, 1, 3, 1, 4), dims=[2]).numpy()
    dim = 2
    eps = 1e-7
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 11
    x1 = (torch.arange(3*2*5, dtype=torch.float64).reshape(3, 2, 5) * 0.01).numpy()
    x2 = (torch.arange(3*2*5, dtype=torch.float64).reshape(3, 2, 5) * 0.01).numpy()
    dim = -1
    eps = 1e-9
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    # Input 12
    small1 = (torch.arange(12, dtype=torch.float32).reshape(4, 3) * 1e-12).numpy()
    small2 = (torch.arange(12, dtype=torch.float32).reshape(4, 3) * 2e-12 + 1e-15).numpy()
    x1 = small1
    x2 = small2
    dim = 1
    eps = 1e-12
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "dim": dim, "eps": eps}))

    return list_of_inputs

generated_inputs["torch.nn.functional.cosine_similarity"] = cosine_similarity_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.cosine_similarity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cosine_similarity'.")


check_valid('torch.nn.functional.cosine_similarity', generated_inputs['torch.nn.functional.cosine_similarity'], lib="torch", suffix=0)
