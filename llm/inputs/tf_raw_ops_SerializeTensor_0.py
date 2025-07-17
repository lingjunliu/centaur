
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SerializeTensor_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensor
    tensor1 = np.array([1, 2, 3], dtype=np.int32)
    tensor1 = tf.constant(tensor1)
    input_dict1 = {"tensor": tensor1, "name": "serialize_int"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor
    tensor2 = np.array([1.0, 2.5, 3.7], dtype=np.float32)
    tensor2 = tf.constant(tensor2)
    input_dict2 = {"tensor": tensor2, "name": "serialize_float"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: String tensor
    tensor3 = np.array(["hello", "world"], dtype=np.string_)
    tensor3 = tf.constant(tensor3)
    input_dict3 = {"tensor": tensor3, "name": "serialize_string"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Boolean tensor
    tensor4 = np.array([True, False, True], dtype=np.bool_)
    tensor4 = tf.constant(tensor4)
    input_dict4 = {"tensor": tensor4, "name": "serialize_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D integer tensor
    tensor5 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor5 = tf.constant(tensor5)
    input_dict5 = {"tensor": tensor5, "name": "serialize_2d_int"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D float tensor
    tensor6 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    tensor6 = tf.constant(tensor6)
    input_dict6 = {"tensor": tensor6, "name": "serialize_3d_float"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Empty tensor
    tensor7 = np.array([], dtype=np.int32)
    tensor7 = tf.constant(tensor7)
    input_dict7 = {"tensor": tensor7, "name": "serialize_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Tensor with negative values
    tensor8 = np.array([-1, -2, 3], dtype=np.int32)
    tensor8 = tf.constant(tensor8)
    input_dict8 = {"tensor": tensor8, "name": "serialize_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SerializeTensor"] = tf_raw_ops_SerializeTensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SerializeTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SerializeTensor'.")

check_valid('tf.raw_ops.SerializeTensor', generated_inputs['tf.raw_ops.SerializeTensor'], lib="tf", suffix=0)
