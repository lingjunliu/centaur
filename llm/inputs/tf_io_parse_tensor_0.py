
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_parse_tensor_inputs():
    list_of_inputs = []

    # Input 1: Simple int32 tensor
    tensor_np = np.array([1, 2, 3], dtype=np.int32)
    tensor_proto = tf.make_tensor_proto(tensor_np)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.int32, "name": "simple_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 tensor with shape
    tensor_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor_proto = tf.make_tensor_proto(tensor_np)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.float32, "name": "float32_shaped"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String tensor
    tensor_np = np.array([b"hello", b"world"])
    tensor_proto = tf.make_tensor_proto(tensor_np, dtype=tf.string)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.string, "name": "string_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Bool tensor
    tensor_np = np.array([True, False, True], dtype=np.bool_)
    tensor_proto = tf.make_tensor_proto(tensor_np)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.bool, "name": "bool_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int64 tensor with negative values
    tensor_np = np.array([-1, 0, 1], dtype=np.int64)
    tensor_proto = tf.make_tensor_proto(tensor_np)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.int64, "name": "int64_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64 tensor
    tensor_np = np.array([1+1j, 2+2j], dtype=np.complex64)
    tensor_proto = tf.make_tensor_proto(tensor_np)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.complex64, "name": "complex64_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rank 3 tensor
    tensor_np = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    tensor_proto = tf.make_tensor_proto(tensor_np)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.int32, "name": "rank3_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tensor
    tensor_np = np.array([], dtype=np.float32)
    tensor_proto = tf.make_tensor_proto(tensor_np)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.float32, "name": "empty_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint8 tensor
    tensor_np = np.array([1, 2, 255], dtype=np.uint8)
    tensor_proto = tf.make_tensor_proto(tensor_np)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.uint8, "name": "uint8_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16 tensor
    tensor_np = np.array([1.0, 2.0, 3.0], dtype=np.float16) # Use float16 for bfloat16 approximation
    tensor_proto = tf.make_tensor_proto(tensor_np, dtype=tf.bfloat16)
    serialized_tensor = tensor_proto.SerializeToString()
    input_dict = {"serialized": serialized_tensor, "out_type": tf.bfloat16, "name": "bfloat16_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.parse_tensor"] = tf_io_parse_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.parse_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.parse_tensor'.")

check_valid('tf.io.parse_tensor', generated_inputs['tf.io.parse_tensor'], lib="tf", suffix=0)
