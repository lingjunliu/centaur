
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rsub_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input1 = torch.tensor([1.0, 2.0, 3.0])
    other1 = torch.tensor([4.0, 5.0, 6.0])
    alpha1 = 1.0
    out1 = torch.tensor([0.0, 0.0, 0.0])

    input_dict1 = {"input": input1.numpy(), "other": other1.numpy(), "alpha": alpha1, "out": out1.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different alpha
    input2 = torch.tensor([1.0, 2.0, 3.0])
    other2 = torch.tensor([4.0, 5.0, 6.0])
    alpha2 = 0.5
    out2 = torch.tensor([0.0, 0.0, 0.0])

    input_dict2 = {"input": input2.numpy(), "other": other2.numpy(), "alpha": alpha2, "out": out2.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0])
    other3 = torch.tensor([-4.0, -5.0, -6.0])
    alpha3 = 1.0
    out3 = torch.tensor([0.0, 0.0, 0.0])

    input_dict3 = {"input": input3.numpy(), "other": other3.numpy(), "alpha": alpha3, "out": out3.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D arrays
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    other4 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    alpha4 = 1.0
    out4 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

    input_dict4 = {"input": input4.numpy(), "other": other4.numpy(), "alpha": alpha4, "out": out4.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different dtype
    input5 = torch.tensor([1, 2, 3], dtype=torch.int32)
    other5 = torch.tensor([4, 5, 6], dtype=torch.int32)
    alpha5 = 1.0
    out5 = torch.tensor([0, 0, 0], dtype=torch.int32)

    input_dict5 = {"input": input5.numpy(), "other": other5.numpy(), "alpha": alpha5, "out": out5.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D array
    input6 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other6 = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    alpha6 = 1.0
    out6 = torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])

    input_dict6 = {"input": input6.numpy(), "other": other6.numpy(), "alpha": alpha6, "out": out6.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: alpha = 0
    input7 = torch.tensor([1.0, 2.0, 3.0])
    other7 = torch.tensor([4.0, 5.0, 6.0])
    alpha7 = 0.0
    out7 = torch.tensor([0.0, 0.0, 0.0])

    input_dict7 = {"input": input7.numpy(), "other": other7.numpy(), "alpha": alpha7, "out": out7.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: larger alpha value
    input8 = torch.tensor([1.0, 2.0, 3.0])
    other8 = torch.tensor([4.0, 5.0, 6.0])
    alpha8 = 2.0
    out8 = torch.tensor([0.0, 0.0, 0.0])

    input_dict8 = {"input": input8.numpy(), "other": other8.numpy(), "alpha": alpha8, "out": out8.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: different shape
    input9 = torch.tensor([[1.0, 2.0, 3.0]])
    other9 = torch.tensor([[4.0, 5.0, 6.0]])
    alpha9 = 1.0
    out9 = torch.tensor([[0.0, 0.0, 0.0]])
    input_dict9 = {"input": input9.numpy(), "other": other9.numpy(), "alpha": alpha9, "out": out9.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: float64
    input10 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    other10 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
    alpha10 = 1.0
    out10 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)

    input_dict10 = {"input": input10.numpy(), "other": other10.numpy(), "alpha": alpha10, "out": out10.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.rsub"] = rsub_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rsub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rsub'.")

check_valid('torch.rsub', generated_inputs['torch.rsub'], lib="torch", suffix=0)
