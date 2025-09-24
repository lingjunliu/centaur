
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def quantize_per_channel_inputs():
    list_of_inputs = []

    # The error "No inputs were generated" indicates that an empty list is not acceptable.
    # Previous errors show that the testing harness cannot convert quantized tensors (the output of this API)
    # to numpy arrays. This creates a contradiction: any valid input will produce an output that crashes
    # the harness. Since an empty list is not allowed, the only remaining option is to provide valid inputs
    # as requested by the prompt, even though they are expected to fail in the test environment.
    # This attempt provides a set of simple, valid inputs covering all supported dtypes.

    # Input 1: Basic 2D tensor, axis=0, quint8
    input_1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    scales_1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_points_1 = np.array([10, 20, 30], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_1,
        'scales': scales_1,
        'zero_points': zero_points_1,
        'axis': 0,
        'dtype': torch.quint8
    }))

    # Input 2: 2D tensor, axis=1, qint8
    input_2 = np.array([[-1.0, 2.5, -3.0], [4.0, -5.5, 6.0]], dtype=np.float32)
    scales_2 = np.array([0.05, 0.1, 0.15], dtype=np.float32)
    zero_points_2 = np.array([-5, 0, 5], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_2,
        'scales': scales_2,
        'zero_points': zero_points_2,
        'axis': 1,
        'dtype': torch.qint8
    }))

    # Input 3: 4D tensor (e.g., conv weight), axis=0, qint8
    input_3 = np.random.randn(4, 2, 3, 3).astype(np.float32)
    scales_3 = (np.random.rand(4) * 0.1 + 1e-5).astype(np.float32)
    zero_points_3 = np.random.randint(-10, 10, size=4, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_3,
        'scales': scales_3,
        'zero_points': zero_points_3,
        'axis': 0,
        'dtype': torch.qint8
    }))

    # Input 4: 4D tensor, axis=1, quint8
    input_4 = np.random.rand(2, 3, 5, 5).astype(np.float32) * 10
    scales_4 = (np.random.rand(3) * 0.2 + 1e-5).astype(np.float32)
    zero_points_4 = np.random.randint(0, 20, size=3, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_4,
        'scales': scales_4,
        'zero_points': zero_points_4,
        'axis': 1,
        'dtype': torch.quint8
    }))

    # Input 5: 3D tensor, axis=2, qint32
    input_5 = (np.random.randn(2, 3, 4) * 1000).astype(np.float32)
    scales_5 = np.array([1e-4, 2e-4, 3e-4, 4e-4], dtype=np.float32)
    zero_points_5 = np.array([0, 0, 0, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_5,
        'scales': scales_5,
        'zero_points': zero_points_5,
        'axis': 2,
        'dtype': torch.qint32
    }))

    # Input 6: Negative axis (-1), 4D tensor, qint8
    input_6 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    scales_6 = (np.random.rand(5) + 1e-6).astype(np.float32)
    zero_points_6 = np.random.randint(-64, 64, size=5, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_6,
        'scales': scales_6,
        'zero_points': zero_points_6,
        'axis': -1,
        'dtype': torch.qint8
    }))

    # Input 7: 1D input tensor, qint8
    input_7 = np.arange(-5, 5, dtype=np.float32)
    scales_7 = (np.random.rand(10) * 0.5 + 1e-6).astype(np.float32)
    zero_points_7 = np.zeros(10, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_7,
        'scales': scales_7,
        'zero_points': zero_points_7,
        'axis': 0,
        'dtype': torch.qint8
    }))
    
    # Input 8: 1D input tensor, quint8
    input_8 = np.arange(0, 10, dtype=np.float32)
    scales_8 = (np.random.rand(10) * 0.5 + 1e-6).astype(np.float32)
    zero_points_8 = np.random.randint(0, 128, size=10, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_8,
        'scales': scales_8,
        'zero_points': zero_points_8,
        'axis': 0,
        'dtype': torch.quint8
    }))

    # Input 9: 1D input tensor, qint32
    input_9 = (np.arange(-5, 5, dtype=np.float32)) * 1000
    scales_9 = (np.random.rand(10) * 1e-3).astype(np.float32)
    zero_points_9 = np.zeros(10, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_9,
        'scales': scales_9,
        'zero_points': zero_points_9,
        'axis': 0,
        'dtype': torch.qint32
    }))

    # Input 10: Larger tensor with qint32
    input_10 = (np.random.randn(8, 2, 2) * 1000).astype(np.float32)
    scales_10 = (np.random.rand(8) * 1e-4).astype(np.float32)
    zero_points_10 = np.zeros(8, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({
        'input': input_10,
        'scales': scales_10,
        'zero_points': zero_points_10,
        'axis': 0,
        'dtype': torch.qint32
    }))
    
    return list_of_inputs

generated_inputs["torch.quantize_per_channel"] = quantize_per_channel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.quantize_per_channel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantize_per_channel'.")

check_valid('torch.quantize_per_channel', generated_inputs['torch.quantize_per_channel'], lib="torch", suffix=0)
