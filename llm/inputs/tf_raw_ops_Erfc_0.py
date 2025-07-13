
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_erfc_inputs():
    list_of_inputs = []

    # Input 1: float32 tensor
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32), "name": "erfc_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor
    x = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x, dtype=tf.float64), "name": "erfc_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 tensor with shape (3, 3)
    x = np.array([[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32), "name": "erfc_float32_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 tensor with shape (2, 2, 2)
    x = np.array([[[ -1.0, -2.0 ], [ -3.0, -4.0 ]], [[ 1.0, 2.0 ], [ 3.0, 4.0 ]]], dtype=np.float64)
    input_dict = {"x": tf.constant(x, dtype=tf.float64), "name": "erfc_float64_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 tensor with shape (1)
    x = np.array([-10.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32), "name": "erfc_float32_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 tensor with shape (1, 1, 1)
    x = np.array([[[ -10.0 ]]], dtype=np.float64)
    input_dict = {"x": tf.constant(x, dtype=tf.float64), "name": "erfc_float64_1_1_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 tensor
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32), "name": "erfc_float32_pos"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 tensor
    x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"x": tf.constant(x, dtype=tf.float64), "name": "erfc_float64_pos"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 tensor with shape (2, )
    x = np.array([-1.0, 1.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x, dtype=tf.float32), "name": "erfc_float32_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float64 tensor with shape (3, )
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x, dtype=tf.float64), "name": "erfc_float64_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Erfc"] = tf_raw_ops_erfc_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Erfc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Erfc'.")

check_valid('tf.raw_ops.Erfc', generated_inputs['tf.raw_ops.Erfc'], lib="tf", suffix=0)
