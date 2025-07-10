
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fft_inputs():
    list_of_inputs = []

    def to_numpy(tensor):
        return tensor.numpy()

    # Input 1: Basic complex64 tensor
    input_tensor = tf.constant(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic complex128 tensor
    input_tensor = tf.constant(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128))
    input_dict = {"input": input_tensor, "name": "fft_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D complex64 tensor
    input_tensor = tf.constant(np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64))
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D complex128 tensor
    input_tensor = tf.constant(np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128))
    input_dict = {"input": input_tensor, "name": "complex_fft"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D complex64 tensor
    input_tensor = tf.constant(np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64))
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Complex64 with negative real and imaginary parts
    input_tensor = tf.constant(np.array([-1-1j, -2-2j, -3-3j], dtype=np.complex64))
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex128 with imaginary part only
    input_tensor = tf.constant(np.array([0+1j, 0+2j, 0+3j], dtype=np.complex128))
    input_dict = {"input": input_tensor, "name": "imag_fft"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.fft"] = tf_signal_fft_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.fft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.fft'.")

check_valid('tf.signal.fft', generated_inputs['tf.signal.fft'], lib="tf", suffix=0)
