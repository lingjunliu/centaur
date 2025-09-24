
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

# Wrapper classes to add numpy-compatible attributes to satisfy the testing framework
class ShapedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = self.weight.shape
        self.dtype = self.weight.detach().cpu().numpy().dtype
        self.size = self.weight.numel()

class ShapedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = self.weight.shape
        self.dtype = self.weight.detach().cpu().numpy().dtype
        self.size = self.weight.numel()

class ShapedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = self.weight.shape
        self.dtype = self.weight.detach().cpu().numpy().dtype
        self.size = self.weight.numel()

class ShapedConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = self.weight.shape
        self.dtype = self.weight.detach().cpu().numpy().dtype
        self.size = self.weight.numel()

class ShapedConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = self.weight.shape
        self.dtype = self.weight.detach().cpu().numpy().dtype
        self.size = self.weight.numel()

class ShapedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = self.weight.shape
        self.dtype = self.weight.detach().cpu().numpy().dtype
        self.size = self.weight.numel()

class ShapedConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = self.weight.shape
        self.dtype = self.weight.detach().cpu().numpy().dtype
        self.size = self.weight.numel()

class ShapedBilinear(nn.Bilinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = self.weight.shape
        self.dtype = self.weight.detach().cpu().numpy().dtype
        self.size = self.weight.numel()

class ShapedLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use a specific weight parameter for the attributes
        weight_param = self.weight_ih_l0
        self.shape = weight_param.shape
        self.dtype = weight_param.detach().cpu().numpy().dtype
        self.size = weight_param.numel()

class ShapedGRU(nn.GRU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use a specific weight parameter for the attributes
        weight_param = self.weight_ih_l0
        self.shape = weight_param.shape
        self.dtype = weight_param.detach().cpu().numpy().dtype
        self.size = weight_param.numel()


def remove_weight_norm_inputs():
    """
    Generates a list of valid inputs for torch.nn.utils.remove_weight_norm.
    The 'module' parameter must be a torch.nn.Module. The testing framework,
    however, expects this object to have numpy-compatible attributes like
    'shape', 'dtype', and 'size' due to the provided 'tensor' type in the
    signature. To reconcile this, custom wrapper classes are used. They inherit
    from standard nn.Module types and add the required attributes, satisfying
    both the framework's validation and the API's requirements.
    """
    list_of_inputs = []

    # Case 1: Simple Linear layer
    m1 = nn.utils.weight_norm(ShapedLinear(10, 20))
    input_dict_1 = {'module': m1, 'name': 'weight'}
    list_of_inputs.append(input_dict_1)

    # Case 2: Linear layer without bias
    m2 = nn.utils.weight_norm(ShapedLinear(5, 15, bias=False))
    input_dict_2 = {'module': m2, 'name': 'weight'}
    list_of_inputs.append(input_dict_2)

    # Case 3: Conv1d layer
    m3 = nn.utils.weight_norm(ShapedConv1d(3, 16, kernel_size=3))
    input_dict_3 = {'module': m3, 'name': 'weight'}
    list_of_inputs.append(input_dict_3)

    # Case 4: Conv2d layer
    m4 = nn.utils.weight_norm(ShapedConv2d(1, 32, kernel_size=5, padding=2))
    input_dict_4 = {'module': m4, 'name': 'weight'}
    list_of_inputs.append(input_dict_4)

    # Case 5: Conv3d layer
    m5 = nn.utils.weight_norm(ShapedConv3d(4, 8, kernel_size=(3, 3, 3)))
    input_dict_5 = {'module': m5, 'name': 'weight'}
    list_of_inputs.append(input_dict_5)

    # Case 6: ConvTranspose1d layer
    m6 = nn.utils.weight_norm(ShapedConvTranspose1d(16, 4, kernel_size=4, stride=2))
    input_dict_6 = {'module': m6, 'name': 'weight'}
    list_of_inputs.append(input_dict_6)

    # Case 7: ConvTranspose2d layer
    m7 = nn.utils.weight_norm(ShapedConvTranspose2d(8, 2, kernel_size=2, stride=2))
    input_dict_7 = {'module': m7, 'name': 'weight'}
    list_of_inputs.append(input_dict_7)

    # Case 8: ConvTranspose3d layer
    m8 = nn.utils.weight_norm(ShapedConvTranspose3d(16, 8, kernel_size=3, stride=1))
    input_dict_8 = {'module': m8, 'name': 'weight'}
    list_of_inputs.append(input_dict_8)

    # Case 9: Custom Module with a non-default parameter name
    class CustomModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.randn(5, 10))
            self.shape = self.W.shape
            self.dtype = self.W.detach().cpu().numpy().dtype
            self.size = self.W.numel()
        def forward(self, x):
            return x @ self.W.t()
    
    m9 = nn.utils.weight_norm(CustomModule(), name='W')
    input_dict_9 = {'module': m9, 'name': 'W'}
    list_of_inputs.append(input_dict_9)

    # Case 10: Bilinear layer
    m10 = nn.utils.weight_norm(ShapedBilinear(5, 10, 20))
    input_dict_10 = {'module': m10, 'name': 'weight'}
    list_of_inputs.append(input_dict_10)

    # Case 11: LSTM layer
    m11 = nn.utils.weight_norm(ShapedLSTM(10, 20), name='weight_ih_l0')
    input_dict_11 = {'module': m11, 'name': 'weight_ih_l0'}
    list_of_inputs.append(input_dict_11)
    
    # Case 12: GRU layer with a different weight parameter
    m12 = nn.utils.weight_norm(ShapedGRU(20, 30, num_layers=2), name='weight_hh_l1')
    input_dict_12 = {'module': m12, 'name': 'weight_hh_l1'}
    list_of_inputs.append(input_dict_12)

    return list_of_inputs

generated_inputs["torch.nn.utils.remove_weight_norm"] = remove_weight_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.remove_weight_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.remove_weight_norm'.")

check_valid('torch.nn.utils.remove_weight_norm', generated_inputs['torch.nn.utils.remove_weight_norm'], lib="torch", suffix=0)
