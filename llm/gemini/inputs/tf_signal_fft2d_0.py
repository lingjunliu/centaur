
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fft2d_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D complex tensor
    input_tensor = tf.constant(np.array([[1+1j, 2+2j], [3+3j, 4+4j]]), dtype=tf.complex64)
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D complex tensor
    input_tensor = tf.constant(np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]]), dtype=tf.complex64)
    name = "fft2d_3d"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different complex values
    input_tensor = tf.constant(np.array([[1-1j, 2+0j], [0+3j, 4-2j]]), dtype=tf.complex64)
    name = ""
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger tensor
    input_tensor = tf.constant(np.random.rand(4, 4) + 1j * np.random.rand(4, 4), dtype=tf.complex64)
    name = "large_fft"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with only real part
    input_tensor = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.complex64)
    name = None
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with only imaginary part
    input_tensor = tf.constant(np.array([[1j, 2j], [3j, 4j]]), dtype=tf.complex64)
    name = "imaginary_only"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: tf.complex128 tensor
    input_tensor = tf.constant(np.array([[1+1j, 2+2j], [3+3j, 4+4j]]), dtype=tf.complex128)
    name = "complex128"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values
    input_tensor = tf.constant(np.array([[-1-1j, -2+2j], [-3+0j, -4-2j]]), dtype=tf.complex64)
    name = "negative"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A 4D tensor
    input_tensor = tf.constant(np.random.rand(2, 2, 3, 3) + 1j * np.random.rand(2, 2, 3, 3), dtype=tf.complex64)
    name = "4d_tensor"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A tensor with a shape where inner dimensions are not the same
    input_tensor = tf.constant(np.random.rand(2, 4, 3) + 1j * np.random.rand(2, 4, 3), dtype=tf.complex64)
    name = "unequal_dims"
    input_dict = {"input": input_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Convert tf.Tensor to numpy array
    for input_dict in list_of_inputs:
        input_dict["input"] = input_dict["input"].numpy()

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.fft2d"] = tf_signal_fft2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.fft2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.fft2d'.")

check_valid('tf.signal.fft2d', generated_inputs['tf.signal.fft2d'], lib="tf", suffix=0)
