
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_real_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "real_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([-1-2j, -3-4j, -5-6j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "real_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([1.5+2.5j, 3.5+4.5j, 5.5+6.5j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "real_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": "real_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([-1-2j, -3-4j, -5-6j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": "real_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1.5+2.5j, 3.5+4.5j, 5.5+6.5j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": "real_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "real_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([0+0j, 0+0j, 0+0j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": "real_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "real_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[1+1j, 2-2j], [3+3j, 4-4j]], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "real_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
