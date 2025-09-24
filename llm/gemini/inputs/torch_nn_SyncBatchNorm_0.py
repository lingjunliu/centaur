
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy

def syncbatchnorm_inputs():
    list_of_inputs = []

    # The error "returns a function, but the input does not have inner values"
    # combined with the signature now explicitly requiring an 'input' tensor,
    # suggests the test harness separates constructor arguments from the forward
    # pass argument ('input'). The failure is likely due to SyncBatchNorm's
    # requirement for an initialized distributed process group, which the test
    # environment may lack, causing the forward pass to fail internally. The
    # harness then reports its generic error.
    # The fix is to provide a well-formed dictionary that satisfies the complete
    # signature, which is the best that can be done without controlling the
    # execution environment.

    # Input 1: Basic case for 2D data
    input_1 = {
        'input': numpy.random.randn(10, 20).astype(numpy.float32),
        'num_features': 20,
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': True,
        'track_running_stats': True,
        'process_group': [],
        'dtype': numpy.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: 3D data, no affine parameters, float64
    input_2 = {
        'input': numpy.random.randn(4, 64, 16).astype(numpy.float64),
        'num_features': 64,
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': False,
        'track_running_stats': True,
        'process_group': [],
        'dtype': numpy.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: 4D data, no tracking of running stats
    input_3 = {
        'input': numpy.random.randn(8, 128, 32, 32).astype(numpy.float32),
        'num_features': 128,
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': True,
        'track_running_stats': False,
        'process_group': [],
        'dtype': numpy.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: 5D data, different numerical parameters
    input_4 = {
        'input': numpy.random.randn(2, 256, 8, 8, 8).astype(numpy.float32),
        'num_features': 256,
        'eps': 1e-04,
        'momentum': 0.05,
        'affine': True,
        'track_running_stats': True,
        'process_group': [],
        'dtype': numpy.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Both affine and track_running_stats are False
    input_5 = {
        'input': numpy.random.randn(4, 512, 16, 16).astype(numpy.float64),
        'num_features': 512,
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': False,
        'track_running_stats': False,
        'process_group': [],
        'dtype': numpy.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_5))
    
    # Input 6: Large number of features
    input_6 = {
        'input': numpy.random.randn(1, 1024).astype(numpy.float32),
        'num_features': 1024,
        'eps': 1e-06,
        'momentum': 0.9,
        'affine': True,
        'track_running_stats': True,
        'process_group': [],
        'dtype': numpy.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_6))
    
    # Input 7: Small number of features and different momentum
    input_7 = {
        'input': numpy.random.randn(16, 16, 64, 64).astype(numpy.float32),
        'num_features': 16,
        'eps': 1e-05,
        'momentum': 0.2,
        'affine': True,
        'track_running_stats': True,
        'process_group': [],
        'dtype': numpy.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Low momentum
    input_8 = {
        'input': numpy.random.randn(20, 32, 50).astype(numpy.float64),
        'num_features': 32,
        'eps': 1e-05,
        'momentum': 0.01,
        'affine': True,
        'track_running_stats': True,
        'process_group': [],
        'dtype': numpy.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_8))
    
    # Input 9: High momentum and no affine
    input_9 = {
        'input': numpy.random.randn(3, 8, 12, 12, 12).astype(numpy.float32),
        'num_features': 8,
        'eps': 1e-05,
        'momentum': 0.99,
        'affine': False,
        'track_running_stats': True,
        'process_group': [],
        'dtype': numpy.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_9))
    
    # Input 10: Larger eps for numerical stability
    input_10 = {
        'input': numpy.random.randn(5, 4, 10).astype(numpy.float32),
        'num_features': 4,
        'eps': 0.001,
        'momentum': 0.1,
        'affine': True,
        'track_running_stats': True,
        'process_group': [],
        'dtype': numpy.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["torch.nn.SyncBatchNorm"] = syncbatchnorm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.SyncBatchNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SyncBatchNorm'.")

check_valid('torch.nn.SyncBatchNorm', generated_inputs['torch.nn.SyncBatchNorm'], lib="torch", suffix=0)
