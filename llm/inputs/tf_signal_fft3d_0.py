
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fft3d_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D complex64 tensor
    input_tensor = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    input_dict = {"input": input_tensor.astype(np.complex64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D complex128 tensor
    input_tensor = np.random.rand(1, 2, 3, 4) + 1j * np.random.rand(1, 2, 3, 4)
    input_dict = {"input": input_tensor.astype(np.complex128), "name": "fft_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 5D complex64 tensor
    input_tensor = np.random.rand(1, 1, 2, 3, 4) + 1j * np.random.rand(1, 1, 2, 3, 4)
    input_dict = {"input": input_tensor.astype(np.complex64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D complex128 tensor with smaller dimensions
    input_tensor = np.random.rand(1, 2, 2) + 1j * np.random.rand(1, 2, 2)
    input_dict = {"input": input_tensor.astype(np.complex128), "name": "small_fft"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D complex64 tensor with some values being zero
    input_tensor = np.random.choice([0, 1], size=(2, 3, 4), p=[0.5, 0.5]) + 1j * np.random.choice([0, 1], size=(2, 3, 4), p=[0.5, 0.5])
    input_dict = {"input": input_tensor.astype(np.complex64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D complex64 tensor with different shape
    input_tensor = np.random.rand(2, 2, 2, 2) + 1j * np.random.rand(2, 2, 2, 2)
    input_dict = {"input": input_tensor.astype(np.complex64), "name": "fft_input_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D complex128 tensor with larger dimensions
    input_tensor = np.random.rand(4, 5, 6) + 1j * np.random.rand(4, 5, 6)
    input_dict = {"input": input_tensor.astype(np.complex128), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D complex64 with specific values
    input_tensor = np.array([[[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]], [[7+7j, 8+8j, 9+9j], [10+10j, 11+11j, 12+12j]]], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D complex128 tensor with specific values
    input_tensor = np.array([[[[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]]]], dtype=np.complex128)
    input_dict = {"input": input_tensor, "name": "complex_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D complex64 tensor
    input_tensor = np.random.rand(2, 1, 2, 3, 4) + 1j * np.random.rand(2, 1, 2, 3, 4)
    input_dict = {"input": input_tensor.astype(np.complex64), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.fft3d"] = tf_signal_fft3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.fft3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.fft3d'.")

check_valid('tf.signal.fft3d', generated_inputs['tf.signal.fft3d'], lib="tf", suffix=0)
