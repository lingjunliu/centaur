
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_shape_inputs():
    list_of_inputs = []

    # Input 1: Scalar input
    input_tensor = tf.constant(1.0).numpy()
    out_type = tf.int32
    name = "scalar_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor
    input_tensor = tf.constant([1, 2, 3, 4, 5]).numpy()
    out_type = tf.int32
    name = "1d_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6]]).numpy()
    out_type = tf.int32
    name = "2d_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    out_type = tf.int32
    name = "3d_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with different data type
    input_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32).numpy()
    out_type = tf.int32
    name = "float_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with int64 out_type
    input_tensor = tf.constant([1, 2, 3]).numpy()
    out_type = tf.int64
    name = "int64_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor
    input_tensor = tf.constant([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]]).numpy()
    out_type = tf.int32
    name = "4d_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Empty tensor
    input_tensor = np.array([])
    out_type = tf.int32
    name = "empty_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with a name
    input_tensor = tf.constant([1, 2, 3]).numpy()
    out_type = tf.int32
    name = "named_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with different shape
    input_tensor = tf.constant([[1, 2], [3, 4], [5, 6]]).numpy()
    out_type = tf.int32
    name = "diff_shape"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.shape"] = tf_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.shape'.")

check_valid('tf.shape', generated_inputs['tf.shape'], lib="tf", suffix=0)
