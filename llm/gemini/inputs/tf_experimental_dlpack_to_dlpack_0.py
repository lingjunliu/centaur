
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_dlpack_to_dlpack_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensor
    tf_tensor = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    input_dict = {"tf_tensor": tf_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor
    tf_tensor = tf.constant(np.array([1.0, 2.5, 3.7, 4.2, 5.9], dtype=np.float32))
    input_dict = {"tf_tensor": tf_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer tensor
    tf_tensor = tf.constant(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64))
    input_dict = {"tf_tensor": tf_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float tensor
    tf_tensor = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64))
    input_dict = {"tf_tensor": tf_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative values
    tf_tensor = tf.constant(np.array([-1, -2, 0, 1, 2], dtype=np.int32))
    input_dict = {"tf_tensor": tf_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with different data types (uint8)
    tf_tensor = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.uint8))
    input_dict = {"tf_tensor": tf_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty Tensor
    tf_tensor = tf.constant(np.array([], dtype=np.int32))
    input_dict = {"tf_tensor": tf_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large Tensor
    tf_tensor = tf.constant(np.random.rand(100, 100).astype(np.float32))
    input_dict = {"tf_tensor": tf_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boolean Tensor
    tf_tensor = tf.constant(np.array([True, False, True, True, False], dtype=np.bool_))
    input_dict = {"tf_tensor": tf_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.dlpack.to_dlpack"] = tf_experimental_dlpack_to_dlpack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.dlpack.to_dlpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.dlpack.to_dlpack'.")

check_valid('tf.experimental.dlpack.to_dlpack', generated_inputs['tf.experimental.dlpack.to_dlpack'], lib="tf", suffix=0)
