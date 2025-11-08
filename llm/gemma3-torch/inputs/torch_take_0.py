
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def take_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    index1 = torch.tensor([0, 2]).numpy()
    input_dict1 = {"input": input1, "index": index1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([10, 20, 30, 40, 50]).numpy()
    index2 = torch.tensor([1, 3, 4]).numpy()
    input_dict2 = {"input": input2, "index": index2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(2, 3, 4).numpy()
    index3 = torch.tensor([0, 3, 6, 9]).numpy()
    input_dict3 = {"input": input3, "index": index3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    index4 = torch.tensor([0, 4, 8]).numpy()
    input_dict4 = {"input": input4, "index": index4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    index5 = torch.tensor([-1, 1, 3]).numpy()
    input_dict5 = {"input": input5, "index": index5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(5, 5).numpy()
    index6 = torch.tensor([0, 0, 0, 0, 0]).numpy()
    input_dict6 = {"input": input6, "index": index6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.tensor([[1, 2], [3, 4]]).numpy()
    index7 = torch.tensor([1, 1]).numpy()
    input_dict7 = {"input": input7, "index": index7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.tensor([1, 2, 3]).numpy()
    index8 = torch.tensor([0, 1, 2]).numpy()
    input_dict8 = {"input": input8, "index": index8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(2, 2).numpy()
    index9 = torch.tensor([3]).numpy()
    input_dict9 = {"input": input9, "index": index9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([10, 20, 30]).numpy()
    index10 = torch.tensor([0, 0, 0]).numpy()
    input_dict10 = {"input": input10, "index": index10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.take"] = take_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.take' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.take'.")


check_valid('torch.take', generated_inputs['torch.take'], lib="torch", suffix=0)
