
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floor_inputs():
    list_of_inputs = []

    # Input 1: half scalar
    x = np.array(3.14, dtype=np.float16)
    name = "floor_scalar_half"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: half scalar
    x = np.array(-2.71, dtype=np.float16)
    name = "floor_scalar_half"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 scalar
    x = np.array(0.0, dtype=np.float32)
    name = "floor_scalar_float32"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 scalar
    x = np.array(-1.0, dtype=np.float64)
    name = "floor_scalar_float64"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: half vector
    x = np.array([1.1, 2.2, 3.3], dtype=np.float16)
    name = "floor_vector_half"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half matrix
    x = np.array([[-1.5, 2.5], [3.5, -4.5]], dtype=np.float16)
    name = "floor_matrix_half"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 tensor (3D)
    x = np.array([[[1.7, 2.3], [3.9, 4.1]], [[5.2, 6.8], [7.4, 8.6]]], dtype=np.float32)
    name = "floor_tensor_float32"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 tensor (4D)
    x = np.array([[[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]],
                  [[[9.9, 10.1], [11.2, 12.3]], [[13.4, 14.5], [15.6, 16.7]]]], dtype=np.float64)
    name = "floor_tensor_float64"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float64), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half tensor with negative and positive values
    x = np.array([-1.5, 2.5, -3.5, 4.5], dtype=np.float16)
    name = "floor_neg_pos_half"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half scalar with large value
    x = np.array(10000.0, dtype=np.float16)
    name = "floor_large_half"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float16), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Floor"] = tf_raw_ops_floor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Floor'.")

check_valid('tf.raw_ops.Floor', generated_inputs['tf.raw_ops.Floor'], lib="tf", suffix=0)
