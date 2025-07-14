
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def replicationpad2d_inputs():
    list_of_inputs = []

    # Input 1: int padding
    input1 = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).numpy()
    padding1 = 2
    input_dict1 = {"padding": (padding1,), "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: tuple padding (left, right, top, bottom)
    input2 = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).numpy()
    padding2 = (1, 1, 1, 1)
    input_dict2 = {"padding": padding2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different shaped input
    input3 = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4).numpy()
    padding3 = (1, 1, 1, 1)
    input_dict3 = {"padding": padding3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Input with only 1 channel
    input4 = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).numpy()
    padding4 = (2,)
    input_dict4 = {"padding": (padding4[0],), "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger padding values
    input5 = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2).numpy()
    padding5 = (1, 1, 1, 1)
    input_dict5 = {"padding": padding5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: No padding
    input6 = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).numpy()
    padding6 = (0,0,0,0)
    input_dict6 = {"padding": padding6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Single channel input
    input7 = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2).numpy()
    padding7 = 2
    input_dict7 = {"padding": (padding7,), "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Batch size > 1
    input8 = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2).numpy()
    padding8 = (1, 1, 1, 1)
    input_dict8 = {"padding": padding8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: C, Hin, Win format
    input9 = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3).numpy()
    padding9 = (1, 1, 1, 1)
    input_dict9 = {"padding": padding9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Different data type
    input10 = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3).numpy()
    padding10 = 2
    input_dict10 = {"padding": (padding10,), "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: padding only left and right, odd number input size
    input11 = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5).numpy()
    padding11 = (1, 1, 0, 0)
    input_dict11 = {"padding": padding11, "input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12: padding only top and bottom, odd number input size
    input12 = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5).numpy()
    padding12 = (0, 0, 1, 1)
    input_dict12 = {"padding": padding12, "input": input12}
    list_of_inputs.append(copy.deepcopy(input_dict12))
    
    # Input 13: padding only left
    input13 = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).numpy()
    padding13 = (1, 0, 0, 0)
    input_dict13 = {"padding": padding13, "input": input13}
    list_of_inputs.append(copy.deepcopy(input_dict13))
    
    # Input 14: padding only top
    input14 = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).numpy()
    padding14 = (0, 0, 1, 0)
    input_dict14 = {"padding": padding14, "input": input14}
    list_of_inputs.append(copy.deepcopy(input_dict14))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ReplicationPad2d_2"] = replicationpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReplicationPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad2d_2'.")

check_valid('torch.nn.ReplicationPad2d', generated_inputs['torch.nn.ReplicationPad2d_2'], lib="torch", suffix=2)
