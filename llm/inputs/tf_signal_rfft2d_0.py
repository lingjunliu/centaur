
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_rfft2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with fft_length matching input size
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "basic_case"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: fft_length smaller than input size (cropping)
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "cropping_case"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: fft_length larger than input size (padding)
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "padding_case"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different fft_lengths for each dimension
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    fft_length = np.array([2, 4], dtype=np.int32)
    name = "different_lengths"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using float64
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "float64_case"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D input, rfft2d applies to last 2 dimensions
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    fft_length = np.array([3, 4], dtype=np.int32)
    name = "3d_input"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: 4D input
    input_tensor = np.random.rand(2, 2, 3, 4).astype(np.float32)
    fft_length = np.array([3, 4], dtype=np.int32)
    name = "4d_input"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small values
    input_tensor = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "small_values"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non square matrix
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    fft_length = np.array([2, 3], dtype=np.int32)
    name = "non_square"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: fft_length with larger values than input tensor dimensions
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    fft_length = np.array([5, 6], dtype=np.int32)
    name = "large_fft_length"
    input_dict = {"input_tensor": input_tensor, "fft_length": fft_length, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.rfft2d"] = tf_signal_rfft2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.rfft2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.rfft2d'.")

check_valid('tf.signal.rfft2d', generated_inputs['tf.signal.rfft2d'], lib="tf", suffix=0)
