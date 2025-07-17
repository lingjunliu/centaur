
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ParseTensor_inputs():
    list_of_inputs = []

    # Input 1: Valid scalar string tensor with int32 type
    tensor_proto = tf.make_tensor_proto(values=[1, 2, 3], dtype=tf.int32, shape=[3])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.int32
    input_dict = {"serialized": serialized, "out_type": out_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Valid scalar string tensor with float32 type
    tensor_proto = tf.make_tensor_proto(values=[1.0, 2.0, 3.0], dtype=tf.float32, shape=[3])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.float32
    input_dict = {"serialized": serialized, "out_type": out_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid scalar string tensor with string type
    tensor_proto = tf.make_tensor_proto(values=[b"a", b"b", b"c"], dtype=tf.string, shape=[3])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.string
    input_dict = {"serialized": serialized, "out_type": out_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Valid scalar string tensor with int64 type and multi-dimensional shape
    tensor_proto = tf.make_tensor_proto(values=[1, 2, 3, 4, 5, 6], dtype=tf.int64, shape=[2, 3])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.int64
    input_dict = {"serialized": serialized, "out_type": out_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Valid scalar string tensor with bool type
    tensor_proto = tf.make_tensor_proto(values=[True, False, True], dtype=tf.bool, shape=[3])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.bool
    input_dict = {"serialized": serialized, "out_type": out_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid scalar string tensor with complex64 type
    tensor_proto = tf.make_tensor_proto(values=[1+1j, 2+2j, 3+3j], dtype=tf.complex64, shape=[3])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.complex64
    input_dict = {"serialized": serialized, "out_type": out_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Valid scalar string tensor with float64 type
    tensor_proto = tf.make_tensor_proto(values=[1.1, 2.2, 3.3], dtype=tf.float64, shape=[3])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.float64
    input_dict = {"serialized": serialized, "out_type": out_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Valid scalar string tensor with negative values and int32 type
    tensor_proto = tf.make_tensor_proto(values=[-1, -2, -3], dtype=tf.int32, shape=[3])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.int32
    input_dict = {"serialized": serialized, "out_type": out_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensor
    tensor_proto = tf.make_tensor_proto(values=[], dtype=tf.int32, shape=[0])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.int32
    input_dict = {"serialized": serialized, "out_type": out_type}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: Valid scalar string tensor with uint8 type
    tensor_proto = tf.make_tensor_proto(values=[1, 2, 3], dtype=tf.uint8, shape=[3])
    serialized_tensor = tensor_proto.SerializeToString()
    serialized = tf.constant(serialized_tensor)
    out_type = tf.uint8
    input_dict = {"serialized": serialized, "out_type": out_type}
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
