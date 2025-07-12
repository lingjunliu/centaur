
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_acosh_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([1.0, 2.0, 3.0, 10.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with inf
    x = np.array([1.0, 2.0, np.inf], dtype=np.float64)
    input_dict = {"x": x, "name": "acosh_inf"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"x": x, "name": "acosh_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    x = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=np.complex64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128
    x = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=np.complex128)
    input_dict = {"x": x, "name": "acosh_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: multi-dimensional float32
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 with values close to 1
    x = np.array([1.0, 1.000001, 1.0001, 1.1], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger float64 values
    x = np.array([100.0, 1000.0, 10000.0], dtype=np.float64)
    input_dict = {"x": x, "name": "acosh_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with different shapes
    x = np.arange(1, 28, dtype=np.float32).reshape((3, 3, 3))
    x[0,0,0] = 1.0
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    for i in range(len(list_of_inputs)):
      list_of_inputs[i]['x'] = tf.constant(list_of_inputs[i]['x'])
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Acosh"] = tf_raw_ops_acosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Acosh'.")

check_valid('tf.raw_ops.Acosh', generated_inputs['tf.raw_ops.Acosh'], lib="tf", suffix=0)
