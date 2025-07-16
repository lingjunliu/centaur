
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_angle_inputs():
    list_of_inputs = []

    # Input 1: complex64, default Tout
    input_tensor = np.array([1 + 1j, -2 - 2j, 3 - 3j, -4 + 4j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: complex128, Tout=float64
    input_tensor = np.array([1 + 1j, -2 - 2j, 3 - 3j, -4 + 4j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": "angle_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, Tout=float32, multi-dimensional
    input_tensor = np.array([[1 + 1j, -2 - 2j], [3 - 3j, -4 + 4j]], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "angle_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, Tout=float64, multi-dimensional, zeros and nans
    input_tensor = np.array([[0 + 0j, np.nan + np.nan * 1j], [3 - 3j, -4 + 4j]], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": "angle_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, Tout=float32, negative real and imag parts
    input_tensor = np.array([-1 - 1j, -2 + 2j, -3 - 3j, -4 + 4j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "angle_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, Tout=float64, all positive
    input_tensor = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": "angle_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64, Tout=float32, single element
    input_tensor = np.array([1 + 1j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "angle_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128, Tout=float64, larger tensor
    input_tensor = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
    input_tensor = input_tensor.astype(np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": "angle_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64, Tout=float32, tensor with 0 imag
    input_tensor = np.array([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "angle_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: complex128, Tout=float64, tensor with 0 real
    input_tensor = np.array([0 + 1j, 0 - 2j, 0 + 3j, 0 - 4j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": "angle_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Angle"] = tf_raw_ops_angle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Angle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Angle'.")

check_valid('tf.raw_ops.Angle', generated_inputs['tf.raw_ops.Angle'], lib="tf", suffix=0)
