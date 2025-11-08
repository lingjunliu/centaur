
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def tripletmarginloss_inputs():
    list_of_inputs = []

    # 1
    anchor = torch.randn(100, 128).numpy()
    positive = torch.randn(100, 128).numpy()
    negative = torch.randn(100, 128).numpy()
    input_dict = {
        "margin": 1.0,
        "p": 2,
        "eps": 1e-6,
        "swap": False,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    anchor = torch.randn(16).numpy()
    positive = torch.randn(16).numpy()
    negative = torch.randn(16).numpy()
    input_dict = {
        "margin": 0.5,
        "p": 1,
        "eps": 1e-8,
        "swap": True,
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    anchor = torch.randn(1, 3).numpy()
    positive = torch.randn(1, 3).numpy()
    negative = torch.randn(1, 3).numpy()
    input_dict = {
        "margin": 2.0,
        "p": 3,
        "eps": 1e-5,
        "swap": False,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    anchor = torch.randn(32, 64, dtype=torch.float64).numpy()
    positive = torch.randn(32, 64, dtype=torch.float64).numpy()
    negative = torch.randn(32, 64, dtype=torch.float64).numpy()
    input_dict = {
        "margin": 0.2,
        "p": 2,
        "eps": 1e-7,
        "swap": True,
        "size_average": False,
        "reduce": False,
        "reduction": "mean",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    anchor = torch.randn(10, 10, dtype=torch.float16).numpy()
    positive = torch.randn(10, 10, dtype=torch.float16).numpy()
    negative = torch.randn(10, 10, dtype=torch.float16).numpy()
    input_dict = {
        "margin": 1.5,
        "p": 4,
        "eps": 1e-9,
        "swap": False,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    anchor = torch.rand(5, 7).numpy()
    positive = torch.rand(5, 7).numpy()
    negative = torch.rand(5, 7).numpy()
    input_dict = {
        "margin": 0.01,
        "p": 2,
        "eps": 1e-12,
        "swap": True,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    a = torch.randn(64, 32)
    anchor = a.numpy()
    positive = (a + 0.1 * torch.randn(64, 32)).numpy()
    negative = torch.randn(64, 32).numpy()
    input_dict = {
        "margin": 0.3,
        "p": 2,
        "eps": 1e-6,
        "swap": True,
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    anchor = torch.tensor([[-1.0, 2.0], [3.5, -4.2]]).numpy()
    positive = torch.tensor([[-0.9, 2.1], [3.2, -3.9]]).numpy()
    negative = torch.tensor([[5.0, -6.0], [-7.0, 8.0]]).numpy()
    input_dict = {
        "margin": 3.0,
        "p": 10,
        "eps": 1e-6,
        "swap": True,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    a = torch.linspace(0, 1, steps=384).reshape(3, 128)
    anchor = a.numpy()
    positive = (a + 0.05 * torch.randn(3, 128)).numpy()
    negative = torch.flip(a, dims=[1]).numpy()
    input_dict = {
        "margin": 0.75,
        "p": 2,
        "eps": 1e-5,
        "swap": False,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    anchor = torch.randn(8, 1).numpy()
    positive = torch.randn(8, 1).numpy()
    negative = torch.randn(8, 1).numpy()
    input_dict = {
        "margin": 1.0,
        "p": 1,
        "eps": 1e-4,
        "swap": False,
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    a = torch.randn(4, 4)
    anchor = a.numpy()
    positive = a.clone().numpy()
    negative = (-a).numpy()
    input_dict = {
        "margin": 0.3,
        "p": 2,
        "eps": 1e-6,
        "swap": False,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    anchor = torch.tensor([10.0, -5.0, 2.0, -3.0, 7.0, -1.0], dtype=torch.float64).numpy()
    positive = torch.tensor([9.5, -4.5, 2.5, -2.5, 7.5, -1.5], dtype=torch.float64).numpy()
    negative = torch.tensor([-10.0, 5.0, -2.0, 3.0, -7.0, 1.0], dtype=torch.float64).numpy()
    input_dict = {
        "margin": 0.9,
        "p": 3,
        "eps": 1e-9,
        "swap": True,
        "size_average": False,
        "reduce": False,
        "reduction": "mean",
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.TripletMarginLoss"] = tripletmarginloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.TripletMarginLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.TripletMarginLoss'.")


check_valid('torch.nn.TripletMarginLoss', generated_inputs['torch.nn.TripletMarginLoss'], lib="torch", suffix=0)
