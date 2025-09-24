
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def lerp_inputs():
    list_of_inputs = []

    # Input 1: Basic case with scalar weight
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    end1 = torch.tensor([4.0, 5.0, 6.0]).numpy()
    weight1 = np.array(0.5, dtype=np.float32)
    out1 = torch.tensor([]).numpy()
    input_dict1 = {"input": input1, "end": end1, "weight": weight1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Weight tensor with the same shape
    input2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    end2 = torch.tensor([4.0, 5.0, 6.0]).numpy()
    weight2 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out2 = torch.tensor([]).numpy()
    input_dict2 = {"input": input2, "end": end2, "weight": weight2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Broadcasting weight tensor
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    end3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).numpy()
    weight3 = np.array([0.2, 0.4], dtype=np.float32)
    out3 = torch.tensor([]).numpy()
    input_dict3 = {"input": input3, "end": end3, "weight": weight3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Negative values
    input4 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    end4 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    weight4 = np.array(0.5, dtype=np.float32)
    out4 = torch.tensor([]).numpy()
    input_dict4 = {"input": input4, "end": end4, "weight": weight4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different data type, weight as int
    input5 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    end5 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).numpy()
    weight5 = np.array(1, dtype=np.float32)
    out5 = torch.tensor([]).numpy()
    input_dict5 = {"input": input5, "end": end5, "weight": weight5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Multi-dimensional tensors
    input6 = torch.randn(2, 3, 4).numpy()
    end6 = torch.randn(2, 3, 4).numpy()
    weight6 = np.array(0.3, dtype=np.float32)
    out6 = torch.tensor([]).numpy()
    input_dict6 = {"input": input6, "end": end6, "weight": weight6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Weight tensor with broadcasting to higher dimension
    input7 = torch.randn(2, 3).numpy()
    end7 = torch.randn(2, 3).numpy()
    weight7 = torch.randn(3).numpy().astype(np.float32)
    out7 = torch.tensor([]).numpy()
    input_dict7 = {"input": input7, "end": end7, "weight": weight7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Larger weight values
    input8 = torch.tensor([1.0, 2.0]).numpy()
    end8 = torch.tensor([3.0, 4.0]).numpy()
    weight8 = np.array(2.0, dtype=np.float32)
    out8 = torch.tensor([]).numpy()
    input_dict8 = {"input": input8, "end": end8, "weight": weight8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Weight as a tensor with broadcasting
    input9 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    end9 = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).numpy()
    weight9 = np.array([0.2], dtype=np.float32)
    out9 = torch.tensor([]).numpy()
    input_dict9 = {"input": input9, "end": end9, "weight": weight9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Using a near zero weight
    input10 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    end10 = torch.tensor([4.0, 5.0, 6.0]).numpy()
    weight10 = np.array(0.0001, dtype=np.float32)
    out10 = torch.tensor([]).numpy()
    input_dict10 = {"input": input10, "end": end10, "weight": weight10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: Weight is 1D tensor, input and end are 2D tensors that can broadcast
    input11 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    end11 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]).numpy()
    weight11 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out11 = torch.tensor([]).numpy()
    input_dict11 = {"input": input11, "end": end11, "weight": weight11, "out": out11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.lerp_1"] = lerp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.lerp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lerp_1'.")

check_valid('torch.lerp', generated_inputs['torch.lerp_1'], lib="torch", suffix=1)
