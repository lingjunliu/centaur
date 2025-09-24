
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_real_inputs():
    list_of_inputs = []

    # Input 1: complex64, default Tout
    input1 = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict1 = {"input": input1, "Tout": tf.float32, "name": "real_part1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: complex128, default Tout
    input2 = np.array([-1.5+2.5j, 3.7-4.8j, 0+1j], dtype=np.complex128)
    input_dict2 = {"input": input2, "Tout": tf.float64, "name": "real_part2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: complex64, Tout=tf.float32
    input3 = np.array([1j, -2j, 3j], dtype=np.complex64)
    input_dict3 = {"input": input3, "Tout": tf.float32, "name": "real_part3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: complex128, Tout=tf.float64
    input4 = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=np.complex128)
    input_dict4 = {"input": input4, "Tout": tf.float64, "name": "real_part4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: complex64, 2D array, default Tout
    input5 = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict5 = {"input": input5, "Tout": tf.float32, "name": "real_part5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Real"] = tf_raw_ops_real_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Real'.")

check_valid('tf.raw_ops.Real', generated_inputs['tf.raw_ops.Real'], lib="tf", suffix=0)
