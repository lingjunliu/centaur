
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_parse_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic int32 tensor
    tensor1 = tf.constant([1, 2, 3], dtype=tf.int32)
    serialized1 = tf.io.serialize_tensor(tensor1).numpy().tobytes()
    input_dict1 = {"serialized": serialized1, "out_type": np.int32, "name": "tensor1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float32 tensor
    tensor2 = tf.constant([1.0, 2.5, 3.7], dtype=tf.float32)
    serialized2 = tf.io.serialize_tensor(tensor2).numpy().tobytes()
    input_dict2 = {"serialized": serialized2, "out_type": np.float32, "name": "tensor2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Bool tensor
    tensor3 = tf.constant([True, False, True], dtype=tf.bool)
    serialized3 = tf.io.serialize_tensor(tensor3).numpy().tobytes()
    input_dict3 = {"serialized": serialized3, "out_type": np.bool_, "name": "tensor3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Rank 2 int64 tensor
    tensor5 = tf.constant([[1, 2], [3, 4]], dtype=tf.int64)
    serialized5 = tf.io.serialize_tensor(tensor5).numpy().tobytes()
    input_dict5 = {"serialized": serialized5, "out_type": np.int64, "name": "tensor5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 5: Rank 3 float64 tensor
    tensor6 = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float64)
    serialized6 = tf.io.serialize_tensor(tensor6).numpy().tobytes()
    input_dict6 = {"serialized": serialized6, "out_type": np.float64, "name": "tensor6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 6: Negative values int16
    tensor8 = tf.constant([-1, -2, -3], dtype=tf.int16)
    serialized8 = tf.io.serialize_tensor(tensor8).numpy().tobytes()
    input_dict8 = {"serialized": serialized8, "out_type": np.int16, "name": "tensor8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 7: uint8 tensor
    tensor9 = tf.constant([1, 2, 3], dtype=tf.uint8)
    serialized9 = tf.io.serialize_tensor(tensor9).numpy().tobytes()
    input_dict9 = {"serialized": serialized9, "out_type": np.uint8, "name": "tensor9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 8: complex64 tensor
    tensor10 = tf.constant([1 + 1j, 2 + 2j, 3 + 3j], dtype=tf.complex64)
    serialized10 = tf.io.serialize_tensor(tensor10).numpy().tobytes()
    input_dict10 = {"serialized": serialized10, "out_type": np.complex64, "name": "tensor10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    tensor11 = tf.constant("test_string", dtype=tf.string)
    serialized11 = tf.io.serialize_tensor(tensor11).numpy().tobytes()
    input_dict11 = {"serialized": serialized11, "out_type": np.object_, "name": "tensor11"}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    tensor12 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int8)
    serialized12 = tf.io.serialize_tensor(tensor12).numpy().tobytes()
    input_dict12 = {"serialized": serialized12, "out_type": np.int8, "name": "tensor12"}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.parse_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.parse_tensor'.")

check_valid('tf.io.parse_tensor', generated_inputs['tf.io.parse_tensor'], lib="tf", suffix=0)
