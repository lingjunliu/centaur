
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def margin_ranking_loss_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.5, 0.2, 0.8], dtype=np.float32)
    input2 = np.array([0.1, 0.9, 0.3], dtype=np.float32)
    label = np.array([1, 0, 1], dtype=np.float32)
    margin = 0.2
    reduction = 'mean'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "label": label,
        "margin": margin,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.margin_ranking_loss"] = margin_ranking_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.margin_ranking_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.margin_ranking_loss'.")


check_valid('torch.nn.functional.margin_ranking_loss', generated_inputs['torch.nn.functional.margin_ranking_loss'], lib="torch", suffix=0)
