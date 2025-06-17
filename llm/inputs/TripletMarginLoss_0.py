
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def TripletMarginLoss_inputs():
    list_of_inputs = []

    anchor = torch.randn(100, 128).numpy()
    positive = torch.randn(100, 128).numpy()
    negative = torch.randn(100, 128).numpy()
    input_dict = {
        "margin": 1.0,
        "p": 2,
        "eps": 1e-06,
        "swap": False,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    anchor = torch.randn(50, 64).numpy()
    positive = torch.randn(50, 64).numpy()
    negative = torch.randn(50, 64).numpy()
    input_dict = {
        "margin": 0.5,
        "p": 1,
        "eps": 1e-07,
        "swap": True,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    anchor = torch.randn(20, 32).numpy()
    positive = torch.randn(20, 32).numpy()
    negative = torch.randn(20, 32).numpy()
    input_dict = {
        "margin": 1.5,
        "p": 3,
        "eps": 1e-05,
        "swap": False,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    anchor = torch.randn(5, 8).numpy()
    positive = torch.randn(5, 8).numpy()
    negative = torch.randn(5, 8).numpy()
    input_dict = {
        "margin": 2.0,
        "p": 4,
        "eps": 1e-04,
        "swap": False,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "anchor": anchor,
        "positive": positive,
        "negative": negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.TripletMarginLoss"] = TripletMarginLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.TripletMarginLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.TripletMarginLoss'.")

check_valid('torch.nn.TripletMarginLoss', generated_inputs['torch.nn.TripletMarginLoss'], lib="torch")
