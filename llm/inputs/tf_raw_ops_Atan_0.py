
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_atan_inputs():
    list_of_inputs = []

    # Input 1: float32, single value
    x = np.array(1.0, dtype=np.float32)
    name = "atan_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, multiple values, positive and negative
    x = np.array([-1.0, 0.0, 1.0, 2.0, -2.0], dtype=np.float32)
    name = "atan_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, single value
    x = np.array(1.0, dtype=np.float64)
    name = "atan_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64, single value
    x = np.array(1.0 + 1.0j, dtype=np.complex64)
    name = "atan_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128, single value
    x = np.array(1.0 + 1.0j, dtype=np.complex128)
    name = "atan_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "atan_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64, multi-dimensional array
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    name = "atan_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64, multi-dimensional array
    x = np.array([[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]], dtype=np.complex64)
    name = "atan_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128, multi-dimensional array
    x = np.array([[-1.0 - 1.0j, -2.0 - 2.0j], [-3.0 - 3.0j, -4.0 - 4.0j]], dtype=np.complex128)
    name = "atan_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: half, single value
    x = np.array(1.0, dtype=np.float16)
    name = "atan_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Atan"] = tf_raw_ops_atan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Atan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Atan'.")

check_valid('tf.raw_ops.Atan', generated_inputs['tf.raw_ops.Atan'], lib="tf", suffix=0)
