
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

class MockQuantizedTensor:
    def __init__(self, data, q_scale=1.0, q_zero_point=0, quant_min=0, quant_max=255):
        self.data = data
        self.q_scale = q_scale
        self.q_zero_point = q_zero_point
        self.quant_min = quant_min
        self.quant_max = quant_max

    def numpy(self):
        return self.data

    def __repr__(self):
        return f"MockQuantizedTensor(data={self.data}, q_scale={self.q_scale}, q_zero_point={self.q_zero_point})"

def qfunctional_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor
    input1 = MockQuantizedTensor(np.array([1, 2, 3], dtype=np.int32))
    input_dict = {"inner": {"args": [], "kwargs": {"x": input1.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    input2 = MockQuantizedTensor(np.array([[1, 2], [3, 4]], dtype=np.int32))
    input_dict = {"inner": {"args": [], "kwargs": {"x": input2.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    input3 = MockQuantizedTensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32))
    input_dict = {"inner": {"args": [], "kwargs": {"x": input3.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with negative values
    input4 = MockQuantizedTensor(np.array([-1, 0, 1], dtype=np.int32))
    input_dict = {"inner": {"args": [], "kwargs": {"x": input4.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with different data type
    input5 = MockQuantizedTensor(np.array([1, 2, 3], dtype=np.uint8))
    input_dict = {"inner": {"args": [], "kwargs": {"x": input5.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger tensor
    input6 = MockQuantizedTensor(np.random.randint(0, 255, size=(10, 10), dtype=np.int32))
    input_dict = {"inner": {"args": [], "kwargs": {"x": input6.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with a different scale
    input7 = MockQuantizedTensor(np.array([1, 2, 3], dtype=np.int32), q_scale=0.5)
    input_dict = {"inner": {"args": [], "kwargs": {"x": input7.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with a different zero point
    input8 = MockQuantizedTensor(np.array([1, 2, 3], dtype=np.int32), q_zero_point=128)
    input_dict = {"inner": {"args": [], "kwargs": {"x": input8.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with min/max bounds
    input9 = MockQuantizedTensor(np.array([1, 2, 3], dtype=np.int32), quant_min=10, quant_max=200)
    input_dict = {"inner": {"args": [], "kwargs": {"x": input9.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another 2D Tensor
    input10 = MockQuantizedTensor(np.array([[5, 6], [7, 8]], dtype=np.int32))
    input_dict = {"inner": {"args": [], "kwargs": {"x": input10.data}}}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.quantized.QFunctional"] = qfunctional_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.quantized.QFunctional' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.quantized.QFunctional'.")

check_valid('torch.nn.quantized.QFunctional', generated_inputs['torch.nn.quantized.QFunctional'], lib="torch", suffix=0)
