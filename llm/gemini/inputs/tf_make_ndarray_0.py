
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_make_ndarray_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 tensor
    tensor1 = tf.make_tensor_proto(tf.constant([1, 2, 3], dtype=tf.int32).numpy())
    input_dict1 = {"tensor": tensor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float32 tensor
    tensor2 = tf.make_tensor_proto(tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32).numpy())
    input_dict2 = {"tensor": tensor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D complex64 tensor
    tensor3 = tf.make_tensor_proto(tf.constant([[[1+1j, 2+2j], [3+3j, 4+4j]]], dtype=tf.complex64).numpy())
    input_dict3 = {"tensor": tensor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Empty tensor
    tensor4 = tf.make_tensor_proto(tf.constant(np.array([], dtype=np.int32)).numpy())
    input_dict4 = {"tensor": tensor4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D string tensor
    tensor5 = tf.make_tensor_proto(tf.constant(["a", "b", "c"], dtype=tf.string).numpy())
    input_dict5 = {"tensor": tensor5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D bool tensor
    tensor6 = tf.make_tensor_proto(tf.constant([[True, False], [False, True]], dtype=tf.bool).numpy())
    input_dict6 = {"tensor": tensor6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7:  int64 tensor with negative values
    tensor7 = tf.make_tensor_proto(tf.constant([-1, -2, -3], dtype=tf.int64).numpy())
    input_dict7 = {"tensor": tensor7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8:  4D float64 tensor
    tensor8 = tf.make_tensor_proto(tf.constant([[[[1.0]]]], dtype=tf.float64).numpy())
    input_dict8 = {"tensor": tensor8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9:  uint8 tensor
    tensor9 = tf.make_tensor_proto(tf.constant([1, 2, 3], dtype=tf.uint8).numpy())
    input_dict9 = {"tensor": tensor9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10:  rank 0 tensor (scalar)
    tensor10 = tf.make_tensor_proto(tf.constant(5, dtype=tf.int32).numpy())
    input_dict10 = {"tensor": tensor10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11:  string tensor with unicode characters
    tensor11 = tf.make_tensor_proto(tf.constant(["你好", "世界"], dtype=tf.string).numpy())
    input_dict11 = {"tensor": tensor11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12:  bfloat16 tensor
    tensor12 = tf.make_tensor_proto(tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.bfloat16).numpy())
    input_dict12 = {"tensor": tensor12}
    list_of_inputs.append(copy.deepcopy(input_dict12))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.make_ndarray"] = tf_make_ndarray_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.make_ndarray' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.make_ndarray'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.make_ndarray', generated_inputs['tf.make_ndarray'], lib="tf", suffix=0)
