
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Erfc_inputs():
    list_of_inputs = []

    # Input 1: half
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.half), "name": "erfc_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64
    x = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": "erfc_double"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half, 2D
    x = np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.half), "name": "erfc_half_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, 2D
    x = np.array([[-3.0, -2.0], [-1.0, 0.0]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 2D
    x = np.array([[-4.0, -3.0], [-2.0, -1.0]], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": "erfc_double_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, 3D - Reduced size for safety
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, large values - Reduced magnitude
    x = np.array([10.0, 20.0, 30.0], dtype=np.float16)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.half), "name": "large_half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, small values
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, negative values
    x = np.array([-0.1, -0.2, -0.3], dtype=np.float64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": "negative_double"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Erfc"] = tf_raw_ops_Erfc_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Erfc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Erfc'.")

check_valid('tf.raw_ops.Erfc', generated_inputs['tf.raw_ops.Erfc'], lib="tf", suffix=0)
