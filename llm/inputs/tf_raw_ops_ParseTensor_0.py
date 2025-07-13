
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ParseTensor_inputs():
    list_of_inputs = []

    # Input 1
    tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.float32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tensor = tf.constant([True, False, True], dtype=tf.bool)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.bool_, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    tensor = tf.constant([1, 2, 3], dtype=tf.int64)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.float64, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    tensor = tf.constant([[-1, -2], [-3, -4]], dtype=tf.int32)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.int32)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int32, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    tensor = tf.constant(np.array([1, 2, 3, 4], dtype=np.uint8), dtype=tf.uint8)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.uint8, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    tensor = tf.constant(np.array([1, 2, 3, 4], dtype=np.uint16), dtype=tf.uint16)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.uint16, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
   
    # Input 10
    tensor = tf.constant(1, dtype=tf.int8)
    serialized = tf.io.serialize_tensor(tensor)
    input_dict = {"serialized": serialized.numpy(), "out_type": np.int8, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ParseTensor"] = tf_raw_ops_ParseTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ParseTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParseTensor'.")

check_valid('tf.raw_ops.ParseTensor', generated_inputs['tf.raw_ops.ParseTensor'], lib="tf", suffix=0)
