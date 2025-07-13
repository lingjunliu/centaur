
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_rint_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D array
    x = np.array([[-1.5, 2.5], [3.5, -4.5]], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "rint_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, scalar
    x = np.array(-1.5, dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.bfloat16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 1D array
    x = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.float16), "name": "rint_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 3D array
    x = np.array([[[1.2, 2.8], [3.5, 4.1]], [[5.7, 6.3], [7.9, 8.6]]], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, scalar positive
    x = np.array(5.0, dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, scalar negative
    x = np.array(-7.0, dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16, array
    x = np.array([1.1, -2.2, 3.3, -4.4], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.bfloat16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half, 2D array with zero
    x = np.array([[0.0, 1.6], [-2.3, 4.0]], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.float16), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, 1D array with large values
    x = np.array([1000.5, -2000.5, 3000.1, -4000.9], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "rint_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Rint"] = tf_raw_ops_rint_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Rint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Rint'.")

check_valid('tf.raw_ops.Rint', generated_inputs['tf.raw_ops.Rint'], lib="tf", suffix=0)
