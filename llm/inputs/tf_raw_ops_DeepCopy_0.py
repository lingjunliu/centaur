
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DeepCopy_inputs():
    list_of_inputs = []

    # Input 1: Simple float32 tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "copy_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor
    x = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"x": x, "name": "copy_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String tensor
    x = np.array(["a", "b", "c"], dtype=np.string_)
    input_dict = {"x": x, "name": "copy_string"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean tensor
    x = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"x": x, "name": "copy_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 tensor
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "copy_2d_float"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int64 tensor
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"x": x, "name": "copy_3d_int"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with negative values
    x = np.array([-1, -2, 3], dtype=np.int32)
    input_dict = {"x": x, "name": "copy_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty tensor
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x, "name": "copy_empty"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large tensor
    x = np.random.rand(100, 100).astype(np.float32)
    input_dict = {"x": x, "name": "copy_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with zero values
    x = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"x": x, "name": "copy_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DeepCopy"] = tf_raw_ops_DeepCopy_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DeepCopy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DeepCopy'.")

check_valid('tf.raw_ops.DeepCopy', generated_inputs['tf.raw_ops.DeepCopy'], lib="tf", suffix=0)
