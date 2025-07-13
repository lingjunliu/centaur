
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_angle_inputs():
    list_of_inputs = []

    # Input 1: complex64, default Tout
    input_tensor = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: complex128, Tout=float64
    input_tensor = np.array([-1 - 1j, -2 - 2j, -3 - 3j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: complex64, with name
    input_tensor = np.array([1 - 1j, 2 - 2j, 3 - 3j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": "angle_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex128, multi-dimensional
    input_tensor = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, zero imaginary part
    input_tensor = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128, zero real part
    input_tensor = np.array([0 + 1j, 0 + 2j, 0 + 3j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64, negative real and imaginary parts
    input_tensor = np.array([-1 - 1j, -2 - 2j, -3 - 3j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex128, large values
    input_tensor = np.array([1000 + 1000j, 2000 + 2000j, 3000 + 3000j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex64, small values
    input_tensor = np.array([0.001 + 0.001j, 0.002 + 0.002j, 0.003 + 0.003j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "Tout": tf.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, different signs
    input_tensor = np.array([1 - 1j, -2 + 2j, 3 + 0j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "Tout": tf.float64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_angle_inputs()
generated_inputs["tf.raw_ops.Angle"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.Angle"].append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Angle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Angle'.")

check_valid('tf.raw_ops.Angle', generated_inputs['tf.raw_ops.Angle'], lib="tf", suffix=0)
