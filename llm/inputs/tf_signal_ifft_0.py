
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_ifft_inputs():
    list_of_inputs = []

    # Input 1: Basic complex64 tensor
    input1 = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict1 = {"input": tf.constant(input1), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic complex128 tensor
    input2 = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    input_dict2 = {"input": tf.constant(input2), "name": "ifft_op"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D complex64 tensor
    input3 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict3 = {"input": tf.constant(input3), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D complex128 tensor
    input4 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict4 = {"input": tf.constant(input4), "name": "ifft_op"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D complex64 tensor
    input5 = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict5 = {"input": tf.constant(input5), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D complex128 tensor
    input6 = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex128)
    input_dict6 = {"input": tf.constant(input6), "name": "ifft_op"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: complex64 with negative values
    input7 = np.array([-1-1j, -2-2j, 3+3j], dtype=np.complex64)
    input_dict7 = {"input": tf.constant(input7), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: complex128 with negative values
    input8 = np.array([-1-1j, -2-2j, 3+3j], dtype=np.complex128)
    input_dict8 = {"input": tf.constant(input8), "name": "ifft_op"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: complex64 with zero values
    input9 = np.array([0+0j, 0+0j, 3+3j], dtype=np.complex64)
    input_dict9 = {"input": tf.constant(input9), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: complex128 with zero values
    input10 = np.array([0+0j, 0+0j, 3+3j], dtype=np.complex128)
    input_dict10 = {"input": tf.constant(input10), "name": "ifft_op"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.signal.ifft"] = tf_signal_ifft_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.ifft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.ifft'.")

check_valid('tf.signal.ifft', generated_inputs['tf.signal.ifft'], lib="tf", suffix=0)
