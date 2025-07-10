
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitcast_inputs():
    list_of_inputs = []

    # Input 1: Basic example with uint32 to uint8, using tf.constant, and correct dtype
    input_tensor = tf.constant(np.array([0xffffffff], dtype=np.uint32), dtype=tf.uint32)
    data_type = tf.uint8
    name = "bitcast_example_1"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 to int32
    input_tensor = tf.constant(np.array([1.0, -2.5, 3.75], dtype=np.float32), dtype=tf.float32)
    data_type = tf.int32
    name = "bitcast_example_2"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32 to uint8
    input_tensor = tf.constant(np.array([1, -2, 3], dtype=np.int32), dtype=tf.int32)
    data_type = tf.uint8
    name = "bitcast_example_3"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 to uint8
    input_tensor = tf.constant(np.array([1.0, -2.5, 3.75], dtype=np.float64), dtype=tf.float64)
    data_type = tf.uint8
    name = "bitcast_example_4"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64 to int8
    input_tensor = tf.constant(np.array([1, -2, 3], dtype=np.int64), dtype=tf.int64)
    data_type = tf.int8
    name = "bitcast_example_5"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: bfloat16 to uint8
    input_tensor = tf.constant(np.array([1.0, -2.5], dtype=tf.bfloat16.as_numpy_dtype), dtype=tf.bfloat16)
    data_type = tf.uint8
    name = "bitcast_example_6"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half to uint8
    input_tensor = tf.constant(np.array([1.0, -2.5], dtype=tf.float16.as_numpy_dtype), dtype=tf.float16)
    data_type = tf.uint8
    name = "bitcast_example_7"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64 to uint8
    input_tensor = tf.constant(np.array([1.0 + 1j, -2.5 - 2j], dtype=np.complex64), dtype=tf.complex64)
    data_type = tf.uint8
    name = "bitcast_example_8"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multi-dimensional array (2D)
    input_tensor = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.int32), dtype=tf.int32)
    data_type = tf.uint8
    name = "bitcast_example_9"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: Multi-dimensional array (3D)
    input_tensor = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32), dtype=tf.float32)
    data_type = tf.uint8
    name = "bitcast_example_10"
    input_dict = {"input": input_tensor, "type": data_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.bitcast"] = tf_bitcast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.bitcast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitcast'.")

check_valid('tf.bitcast', generated_inputs['tf.bitcast'], lib="tf", suffix=0)
