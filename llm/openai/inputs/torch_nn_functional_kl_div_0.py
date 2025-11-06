
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def kl_div_inputs():
    list_of_inputs = []

    input_arr = torch.nn.functional.log_softmax(torch.randn(5), dim=-1).numpy()
    target_arr = torch.nn.functional.softmax(torch.randn(5), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "mean", "log_target": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.randn(3, 4), dim=-1).numpy()
    target_arr = torch.nn.functional.softmax(torch.randn(3, 4), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "sum", "log_target": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.randn(4, 7, dtype=torch.float64), dim=-1).numpy()
    target_arr = torch.nn.functional.softmax(torch.randn(4, 7, dtype=torch.float64), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "batchmean", "log_target": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.randn(2, 3, 5), dim=-1).numpy()
    target_arr = torch.nn.functional.softmax(torch.randn(2, 3, 5), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "none", "log_target": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.randn(1, 2, 4), dim=-1).numpy()
    target_arr = torch.nn.functional.log_softmax(torch.randn(1, 2, 4), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "mean", "log_target": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.randn(2, 3, 4, 6), dim=-1).numpy()
    target_arr = torch.nn.functional.softmax(torch.randn(2, 3, 4, 6), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "sum", "log_target": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.zeros(1), dim=-1).numpy()
    target_arr = torch.nn.functional.softmax(torch.zeros(1), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "none", "log_target": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.randn(2, 5), dim=-1).numpy()
    target_arr = torch.nn.functional.log_softmax(torch.randn(2, 5), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "none", "log_target": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.randn(5, 1, 3), dim=-1).numpy()
    target_arr = torch.nn.functional.softmax(torch.randn(5, 1, 3), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "mean", "log_target": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = torch.randn(1, 10) * 5
    input_arr = torch.nn.functional.log_softmax(logits, dim=-1).numpy()
    target_arr = torch.nn.functional.softmax(torch.tensor([[10.0] + [0.1]*9]), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "batchmean", "log_target": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.randn(3, 6, dtype=torch.float64), dim=-1).numpy()
    target_arr = torch.nn.functional.log_softmax(torch.randn(3, 6, dtype=torch.float64), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "sum", "log_target": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.nn.functional.log_softmax(torch.randn(1, 2, 2, 3), dim=-1).numpy()
    target_arr = torch.nn.functional.log_softmax(torch.randn(1, 2, 2, 3), dim=-1).numpy()
    input_dict = {"input": input_arr, "target": target_arr, "reduction": "mean", "log_target": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.kl_div"] = kl_div_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.kl_div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.kl_div'.")


check_valid('torch.nn.functional.kl_div', generated_inputs['torch.nn.functional.kl_div'], lib="torch", suffix=0)
