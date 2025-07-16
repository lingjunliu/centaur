
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DebugGradientRefIdentity_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    input_tensor = tf.Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    input_dict = {"input": input_tensor, "name": "float32_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 tensor
    input_tensor = tf.Variable(np.array([-1, 0, 1], dtype=np.int32))
    input_dict = {"input": input_tensor, "name": "int32_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Bool tensor
    input_tensor = tf.Variable(np.array([True, False, True], dtype=np.bool_))
    input_dict = {"input": input_tensor, "name": "bool_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 tensor
    input_tensor = tf.Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    input_dict = {"input": input_tensor, "name": "float64_tensor_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 tensor
    input_tensor = tf.Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64))
    input_dict = {"input": input_tensor, "name": "int64_tensor_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String tensor
    input_tensor = tf.Variable(np.array(["hello", "world"], dtype=np.string_))
    input_dict = {"input": input_tensor, "name": "string_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Complex64 tensor
    input_tensor = tf.Variable(np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64))
    input_dict = {"input": input_tensor, "name": "complex64_tensor_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0-dimensional tensor (scalar)
    input_tensor = tf.Variable(np.array(10.0, dtype=np.float32))
    input_dict = {"input": input_tensor, "name": "scalar_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative values in int32 tensor
    input_tensor = tf.Variable(np.array([-5, -2, 0, 3], dtype=np.int32))
    input_dict = {"input": input_tensor, "name": "negative_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large values in int64 tensor
    input_tensor = tf.Variable(np.array([2**30, 2**20, 2**10], dtype=np.int64))
    input_dict = {"input": input_tensor, "name": "large_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DebugGradientRefIdentity"] = tf_raw_ops_DebugGradientRefIdentity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DebugGradientRefIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DebugGradientRefIdentity'.")

check_valid('tf.raw_ops.DebugGradientRefIdentity', generated_inputs['tf.raw_ops.DebugGradientRefIdentity'], lib="tf", suffix=0)
